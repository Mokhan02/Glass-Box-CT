"""Reports for a scan: the agent when an LLM is configured, otherwise a deterministic template built from the SAME tool results (so the app works offline and the
numbers still come from, and are checked against, the tools). Every report ends with the research-use disclaimer."""
from __future__ import annotations

import hashlib
import json
import os

from glassbox.agent.agent import DISCLAIMER, REPORT_PROMPT, run_agent
from glassbox.agent.backends import PrecomputedBackend
from glassbox.agent.grounding import check_report
from glassbox.agent.tools import ToolBox, qty


def _q(x) -> str:
    return f"{x['value']:g} {x['unit'].replace('cm3', 'cm3')}" if isinstance(x, dict) and "value" in x else str(x)


def template_report(evidence: list[dict]) -> str:
    ev = {e["tool"]: e["result"] for e in evidence if e.get("ok")}; L = ["**Research report (template: no language model was called).**", ""]
    s = ev.get("segment_ct")
    if s:
        L.append("**Segmentation** [segment_ct]: structures found: " + ", ".join(s["structures_found"]) + ".")
    b = ev.get("body_composition")
    if b:
        if b.get("available"):
            L.append(f"**Body composition at L3** [body_composition]: psoas and paraspinal muscle area {_q(b['psoas_paraspinal_area'])}; psoas area {_q(b['psoas_area'])}; mean muscle density {_q(b['mean_muscle_density'])}. {b['label']}.")
        else:
            L.append(f"**Body composition** [body_composition]: not available, because {b['reason']}.")
    les = ev.get("measure_lesions")
    if les:
        if les.get("standing_limitation"):
            L.append(f"**Limitation** [measure_lesions]: {les['standing_limitation']}")
        if les["n_lesions"]:
            for l in les["lesions"]:
                L.append(f"**Lesion {l['id']}** [measure_lesions]: longest axial diameter {_q(l['longest_axial_diameter'])}, volume {_q(l['volume'])}, mean model probability {l['mean_probability']}; RECIST 1.1 suggestion: {l['recist']}.")
            if "tumour_liver_contrast" in les:
                L.append(f"**Tumour-minus-liver contrast** [measure_lesions]: {_q(les['tumour_liver_contrast'])} ({les['contrast_basis']}; reference liver at least {_q(les['liver_reference_margin'])} from the tumour).")
                if "contrast_caution" in les:
                    L.append(f"**{les['contrast_caution']}** [measure_lesions] {les['contrast_context']['evidence_status']}")
            v = les["validation_context"]
            L.append(f"**Known bias of this measurement** [measure_lesions]: in held-out research data the model's detected lesions measured a median {_q(v['detected_lesion_median_error'])} from the reference ({_q(v['detected_lesion_median_relative_error'])}), "
                     f"and the sum of diameters a median {_q(v['case_sum_of_diameters_median_error'])}; it never overestimated by {_q(v['large_overestimate_threshold'])} or more (count: {v['overestimates_at_or_above_threshold']}). {v['note']}")
        else:
            L.append(f"**Lesions** [measure_lesions]: {les.get('note', 'none predicted')}")
    q = ev.get("quality_audit")
    if q:
        for st in q["structures"]:
            if st.get("assessed"):
                d = ", ".join(f"{x['name']} ({x['pushes_toward']})" for x in st["drivers"][:2])
                L.append(f"**Quality audit, {st['name']}** [quality_audit]: estimated probability that the segmentation is wrong {st['failure_probability']}; flag: {st['flag']}. Main drivers: {d}. {st['caveat']}")
            else:
                L.append(f"**Quality audit, {st['name']}** [quality_audit]: cannot be assessed. {st['reason']}")
        L.append(f"[quality_audit] {q['organs_not_assessed']}")
    return "\n\n".join(L) + "\n\n" + DISCLAIMER


REPORT_VERSION = 2  # bump when the template or the agent prompt changes, so cached reports are rebuilt
RESULT_FILES = ("segmentation.json", "body_composition.json", "lesions.json", "quality_audit.json")


def inputs_sha(demo_dir: str, scan_id: str, llm) -> str:
    """Fingerprint of everything a report is built from: the tool-result files, the report version, and which provider/models would write it."""
    h = hashlib.sha256(f"v{REPORT_VERSION}|{(llm[1].provider_host + llm[1].model + llm[1].fast_model + REPORT_PROMPT) if llm else 'template'}".encode())
    for f in RESULT_FILES:
        p = os.path.join(demo_dir, scan_id, f)
        h.update(open(p, "rb").read() if os.path.exists(p) else b"-")
    return h.hexdigest()[:16]


def make_report(demo_dir: str, scan_id: str, llm=None, refresh: bool = False) -> dict:
    cache = os.path.join(demo_dir, scan_id, "report.json"); sha = inputs_sha(demo_dir, scan_id, llm)
    if os.path.exists(cache) and not refresh:
        cached = json.load(open(cache))
        if cached.get("inputs_sha") == sha:
            return cached  # otherwise the tool results (or report version) changed since it was written: rebuild
    tb = ToolBox(PrecomputedBackend(demo_dir))
    if llm is not None:
        client, cfg = llm
        try:
            out = run_agent(scan_id, client, cfg, tb); out["mode"] = "agent"
        except Exception as e:  # provider down or misconfigured: fall back, and say so
            tb = ToolBox(PrecomputedBackend(demo_dir)); out = _template(tb, scan_id); out["note"] = f"language model unavailable ({type(e).__name__}); template used"
    else:
        out = _template(tb, scan_id)
    out["inputs_sha"] = sha
    json.dump(out, open(cache, "w"), indent=1)
    return out


def _template(tb: ToolBox, scan_id: str) -> dict:
    for t in ("segment_ct", "body_composition", "measure_lesions", "quality_audit"):
        tb.call(t, {"scan_id": scan_id})
    rep = template_report(tb.evidence); chk = check_report(rep, tb.evidence)
    return {"scan_id": scan_id, "report": rep, "evidence": tb.evidence, "mode": "template", "provider_host": None, **chk}
