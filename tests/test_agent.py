import json
from types import SimpleNamespace as NS

from glassbox.agent.agent import DISCLAIMER, run_agent
from glassbox.agent.backends import PrecomputedBackend
from glassbox.agent.config import ConfigError, LLMConfig, from_env
from glassbox.agent.grounding import check_report
from glassbox.agent.tools import ToolBox, qty

CFG = LLMConfig("https://example.test/v1", "SECRET-KEY", "big-model", "small-model")


def _backend(tmp_path):
    d = tmp_path / "scanA"; d.mkdir()
    fx = {"fixture": True}
    (d / "segmentation.json").write_text(json.dumps({**fx, "structures": ["liver", "L3"]}))
    (d / "body_composition.json").write_text(json.dumps({**fx, "l3_slice": 146, "muscle_area": qty(45.8, "cm2"), "mean_density": qty(46.4, "HU")}))
    (d / "lesions.json").write_text(json.dumps({**fx, "prompted_classes": [26], "lesions": [{"organ": "liver", "diameter": qty(23.5, "mm")}]}))
    (d / "quality_audit.json").write_text(json.dumps({**fx, "liver": {"failure_probability": 0.12, "flag": False}}))
    (tmp_path / "features.json").write_text(json.dumps({"fixture": True, "7": {"name": "psoas edge"}}))
    return PrecomputedBackend(str(tmp_path), allow_fixture=True)


class FakeClient:
    """Scripted chat client: first call asks for tools, then plain text; the last call is the report."""
    def __init__(self, report):
        self.calls, self.report, self.chat = [], report, NS(completions=NS(create=self._create))
        self.n = 0

    def _create(self, **kw):
        self.calls.append(kw); self.n += 1
        if "tools" in kw and self.n == 1:
            tc = [NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps({"scan_id": "scanA"}))) for i, n in
                  enumerate(["segment_ct", "body_composition", "measure_lesions", "quality_audit", "no_such_tool"])]
            return NS(choices=[NS(message=NS(content="", tool_calls=tc))])
        if "tools" in kw:
            return NS(choices=[NS(message=NS(content="done", tool_calls=None))])
        return NS(choices=[NS(message=NS(content=self.report, tool_calls=None))])


def test_agent_gathers_then_reports_with_models_split_and_disclaimer(tmp_path):
    tb = ToolBox(_backend(tmp_path))
    client = FakeClient("Muscle area is 45.8 cm² [body_composition]; density 46.4 HU; liver risk 0.12 [quality_audit].")
    out = run_agent("scanA", client, CFG, tb)
    assert [c["model"] for c in client.calls] == ["small-model", "small-model", "big-model"]
    assert out["report"].endswith(DISCLAIMER)
    assert out["unsupported_numbers"] == [] and out["missing_units"] == [] and out["ok"]
    assert out["fixture"] and out["report"].startswith("**TEST FIXTURE DATA")  # made-up data is stamped
    assert out["provider_host"] == "example.test"
    assert any(e["tool"] == "no_such_tool" and not e["ok"] for e in out["evidence"])  # bad tool name did not crash the run


def test_grounding_flags_invented_numbers_and_accepts_percent_of_fraction(tmp_path):
    tb = ToolBox(_backend(tmp_path)); tb.call("body_composition", {"scan_id": "scanA"}); tb.call("quality_audit", {"scan_id": "scanA"})
    assert check_report("Area 45.8 cm2 and 12% risk.", tb.evidence)["ok"]
    assert check_report("Area 52.3 cm2, density 46.4 HU.", tb.evidence)["unsupported_numbers"] == ["52.3 cm2"]


def test_path_traversal_scan_id_is_rejected(tmp_path):
    tb = ToolBox(_backend(tmp_path))
    r = tb.call("segment_ct", {"scan_id": "../secret"})
    assert not r["ok"]


def test_config_never_exposes_key_and_requires_vars(tmp_path, monkeypatch):
    assert "SECRET-KEY" not in repr(CFG)
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_FAST_MODEL"):
        monkeypatch.delenv(k, raising=False)
    try:
        from_env(str(tmp_path / "missing.env")); raise AssertionError("should fail")
    except ConfigError as e:
        assert "LLM_API_KEY" in str(e)
    (tmp_path / ".env").write_text("LLM_BASE_URL=https://x.test/v1\nLLM_API_KEY=k\nLLM_MODEL=m\n")
    cfg = from_env(str(tmp_path / ".env"))
    assert cfg.fast_model == "m" and cfg.provider_host == "x.test"


def test_bare_number_without_its_unit_fails_the_check(tmp_path):
    """The smoke-test failure: the model printed 'Mean muscle HU: 46.41' with no unit."""
    tb = ToolBox(_backend(tmp_path)); tb.call("body_composition", {"scan_id": "scanA"})
    r = check_report("Mean muscle HU: 46.4. Area 45.8 cm2.", tb.evidence)
    assert not r["ok"] and r["unsupported_numbers"] == [] and len(r["missing_units"]) == 1 and "46.4" in r["missing_units"][0]
    assert check_report("Mean density 46.4 HU. Area 45.8 cm2.", tb.evidence)["ok"]
    assert not check_report("Mean density 46.4 mm.", tb.evidence)["ok"]  # wrong unit is also a failure


def test_tool_with_bare_measurement_is_rejected(tmp_path):
    class Bad(PrecomputedBackend):
        def body_composition(self, scan_id): return {"muscle_area": 45.8}  # no unit
    r = ToolBox(Bad(str(tmp_path), allow_fixture=True)).call("body_composition", {"scan_id": "scanA"})
    assert not r["ok"] and "without units" in r["error"]


def test_backend_refuses_made_up_data_and_temp_dirs(tmp_path):
    import pytest
    d = tmp_path / "real"; (d / "s1").mkdir(parents=True)
    (d / "s1" / "segmentation.json").write_text(json.dumps({"fixture": True, "structures": []}))
    r = ToolBox(PrecomputedBackend(str(d))).call("segment_ct", {"scan_id": "s1"})  # fixture flag, default backend
    assert not r["ok"] and "fixture" in r["error"]
    with pytest.raises(PermissionError):
        PrecomputedBackend("/private/tmp/claude-1/scratchpad/demo")  # scratch path refused outright


def test_lesion_tool_must_report_that_it_prompted_the_hepatic_tumour_class(tmp_path):
    class NoPrompt(PrecomputedBackend):
        def measure_lesions(self, scan_id): return {"lesions": []}  # looks like "no tumour" but class 26 was never asked for
    r = ToolBox(NoPrompt(str(tmp_path), allow_fixture=True)).call("measure_lesions", {"scan_id": "scanA"})
    assert not r["ok"] and "not prompted" in r["error"]


def test_segment_everything_excludes_tumour_classes_but_report_prompts_include_them():
    from glassbox.seg.prompts import EVERYTHING_EXCLUDED, everything_prompts, report_prompts
    assert 26 in EVERYTHING_EXCLUDED and 26 not in everything_prompts()
    assert 26 in report_prompts() and 23 in report_prompts()
    assert 129 not in report_prompts() and 130 not in report_prompts()  # not promptable by label; need point prompts
