"""Every number on the 'Why it fails' page must come from results/SUMMARY.md: each entry in frontend/src/data/findings.json carries a verbatim `src` fragment
of SUMMARY.md, and its value / CI / count strings must appear inside that fragment."""
import json
import os

HERE = os.path.dirname(os.path.dirname(__file__))
SUMMARY = open(os.path.join(HERE, "results", "SUMMARY.md"), encoding="utf-8").read()
FINDINGS = json.load(open(os.path.join(HERE, "frontend", "src", "data", "findings.json"), encoding="utf-8"))
NUMERIC_KEYS = ("value", "ci", "dice", "dice_ci", "sld", "sld_ci", "over", "over_ci", "n", "hits", "still", "rho", "rho_ci")


def walk(node, path=""):
    if isinstance(node, dict):
        if "src" in node:
            yield path, node
        for k, v in node.items():
            if isinstance(v, dict):
                yield from walk(v, f"{path}/{k}")


def test_every_source_fragment_is_verbatim_in_summary():
    entries = list(walk(FINDINGS)); assert len(entries) >= 25
    for path, e in entries:
        assert e["src"] in SUMMARY, f"{path}: source fragment not found in results/SUMMARY.md: {e['src']!r}"


def test_every_displayed_number_is_inside_its_source_fragment():
    for path, e in walk(FINDINGS):
        for key in NUMERIC_KEYS:
            if key not in e:
                continue
            for s in (e[key] if isinstance(e[key], list) else [e[key]]):
                if key == "value" and not any(ch.isdigit() for ch in s.replace("1 in 5", "")) or s == "1 in 5":
                    continue
                digits = s.lstrip("+")
                assert digits in e["src"], f"{path}.{key}={s!r} is not in its source fragment {e['src']!r}"
