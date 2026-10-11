"""The audit record for one scan: everything needed to see, after the fact, which model produced which measurement under which settings, how far it was trusted and why.
JSON for machines, a readable PDF for people. The record carries a SHA-256 over its own contents and over each tool-result file it was built from, so later edits are detectable.
Research use only. It documents a measurement's reliability; it is not a diagnosis."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
from datetime import datetime, timezone

from glassbox.app.report import RESULT_FILES, make_report
from glassbox.trust.verdict import assess, sentence

HERE = os.path.dirname(__file__)
RECORD_VERSION = 1
THRESHOLDS = {"default_logit": 0.0, "default_probability": 0.5, "recalibrated_logit": -3.0, "recalibrated_probability": round(1 / (1 + math.exp(3.0)), 4)}


def _load(d: str, name: str):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _sha(path: str) -> str | None:
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def _commit() -> str:
    c = os.environ.get("GLASSBOX_COMMIT")
    if c:
        return c
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "integrity"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_audit(demo_dir: str, scan_id: str, llm=None, now: datetime | None = None) -> dict:
    d = os.path.join(demo_dir, scan_id); rep = make_report(demo_dir, scan_id, llm=llm); les, qa, ph, rc = _load(d, "lesions.json"), _load(d, "quality_audit.json"), _load(d, "phase.json"), _load(d, "recalibrated.json")
    q = qa["structures"][0] if qa else None; trust = assess(les, q, ph) if les else None; idx = _load(demo_dir, "index.json") or {"scans": []}; meta = next((s for s in idx["scans"] if s["id"] == scan_id), {})
    rec = {
        "record_version": RECORD_VERSION, "purpose": "Audit of an AI lesion measurement: what was measured, by which model and settings, and how far it can be trusted. Research use only; not a diagnosis.",
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "scan": {"id": scan_id, "dataset": (meta.get("source") or {}).get("dataset"), "case": (meta.get("source") or {}).get("case"), "license": (meta.get("source") or {}).get("license"), "provenance": (les or {}).get("provenance")},
        "model": json.load(open(os.path.join(HERE, "model_card.json"))), "thresholds": THRESHOLDS,
        "software": {"glassbox_commit": _commit(), "report_version": rep.get("writer_sha") and "see agent.writer_sha"},
        "measurements": {"default": (rc or {}).get("default"), "recalibrated": (rc or {}).get("recalibrated"), "recalibrated_logit": (rc or {}).get("threshold_logit"), "lesions": (les or {}).get("lesions"),
                         "sum_of_target_diameters": (les or {}).get("sum_of_target_diameters"), "tumour_liver_contrast": (les or {}).get("tumour_liver_contrast"), "reference_contrast_hu": meta.get("reference_contrast_hu")},
        "quality_flag": q, "phase": ph or {"estimate": None, "note": "no phase estimate was stored for this scan"},
        "verdict": trust,
        "agent": {"mode": rep.get("mode"), "provider_host": rep.get("provider_host"), "models": rep.get("models"), "attempts": rep.get("attempts"), "report_generated_at": rep.get("generated_at"), "cached": rep.get("cached"),
                  "report": rep.get("report"), "number_check_ok": rep.get("ok"), "unsupported_numbers": rep.get("unsupported_numbers"), "missing_units": rep.get("missing_units"), "writer_sha": rep.get("writer_sha"), "note": rep.get("note")},
        "tool_call_log": [{"step": i + 1, "tool": e["tool"], "args": e.get("args"), "ok": e.get("ok"), "error": e.get("error"), "result": e.get("result")} for i, e in enumerate(rep.get("evidence") or [])],
        "limitations": [(les or {}).get("standing_limitation"), "Verdicts are a research heuristic about the measurement, not a diagnosis.", "The recalibrated threshold trades under-measurement for over-measurement; see results/SUMMARY.md for the measured cost."],
        "inputs_sha256": {f: _sha(os.path.join(d, f)) for f in RESULT_FILES + ("phase.json", "recalibrated.json")},
    }
    rec["integrity"] = {"algorithm": "sha256 over this record without this field (sorted keys, compact JSON)", "sha256": _digest(rec)}
    return rec


def verify_audit(rec: dict) -> bool:
    return rec.get("integrity", {}).get("sha256") == _digest(rec)


_PDF_MAP = {"\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2212": "-", "\u2014": " - ", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2265": ">=", "\u2264": "<=", "\u2192": "->", "\u00a0": " ", "\u2009": " "}


def pdf_safe(text: str) -> str:
    """The PDF uses the built-in Helvetica (Latin-1): map the typographic characters models like to write, replace anything else it cannot draw."""
    for a, b in _PDF_MAP.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def audit_pdf(rec: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape as _esc

    e = lambda x: _esc(pdf_safe(str(x)))  # noqa: E731

    ss = getSampleStyleSheet(); base = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12.5); h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4); small = ParagraphStyle("s", parent=base, fontSize=7.5, leading=10, textColor=colors.HexColor("#555555"))
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=f"Glass-Box CT audit record {rec['scan']['id']}", author="Glass-Box CT")
    st: list = []; v = rec.get("verdict") or {}; lvl = v.get("level", "n/a"); col = {"Trusted": "#117a65", "Review": "#b9770e", "Unreliable": "#c0392b"}.get(lvl, "#333333")
    st += [Paragraph("Glass-Box CT: audit record", ss["Title"]), Paragraph(f"Scan {e(rec['scan']['id'])} &middot; {e(str(rec['scan'].get('dataset')))} {e(str(rec['scan'].get('case')))} &middot; generated {e(rec['generated_at'])}", base),
           Paragraph("Research use only. Not for clinical decisions. This record documents how far an AI measurement can be relied on; it is not a diagnosis.", small),
           Paragraph("Verdict", h), Paragraph(f'<font color="{col}"><b>{e(lvl.upper())}</b></font>: {e(sentence(lvl)) if lvl in ("Trusted", "Review", "Unreliable") else ""}', base)]
    st += [Paragraph(f"&bull; {e(r['text'])} <font size=7 color='#777777'>[source: {e(r['source'])}]</font>", base) for r in v.get("reasons", [])]
    ms = rec["measurements"]; rows = [["", "Default (logit > 0)", "Recalibrated (research)"]]
    dflt, recal = ms.get("default") or {}, ms.get("recalibrated") or {}
    for lab, k in (("Sum of target diameters", "sum_of_diameters"), ("Largest lesion", "largest_diameter"), ("Predicted tumour volume", "volume")):
        f = lambda x: f"{x[k]['value']} {x[k]['unit']}" if x and x.get(k) else "n/a"
        rows.append([lab, f(dflt), f(recal)])
    t = Table(rows, colWidths=[55 * mm, 55 * mm, 55 * mm]); t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb"))]))
    c = ms.get("tumour_liver_contrast"); st += [Paragraph("Measurements: default and recalibrated", h), t, Paragraph(f"Tumour minus liver (predicted tumour): {c['value']} {c['unit']}" if c else "Tumour-minus-liver contrast: not available", base),
           Paragraph(f"The recalibrated view lowers the tumour threshold to logit {THRESHOLDS['recalibrated_logit']:g}; the default stays primary. Cost measured on held-out research data: over-estimates by 20% or more in 13.5% of cases (results/SUMMARY.md).", small)]
    qf = rec.get("quality_flag") or {}; st += [Paragraph("Quality flag", h), Paragraph(f"Flag: <b>{e(str(qf.get('flag')))}</b>; estimated probability that the segmentation is wrong {e(str(qf.get('failure_probability')))} (threshold {e(str(qf.get('flag_threshold_probability')))}). Validated on: {e(str(qf.get('validated_on')))}.", base)]
    st += [Paragraph(f"&bull; {e(d['name'])} pushes toward {e(d['pushes_toward'])} (weight {d['weight_fraction']})", base) for d in (qf.get("drivers") or [])]
    ph = rec["phase"]; hu = lambda k: (f"{ph[k]['value']} HU" if ph.get(k) else "not measurable")  # noqa: E731
    st += [Paragraph("Contrast phase (research heuristic)", h), Paragraph(f"Estimate: <b>{e(ph['estimate'])}</b>; aorta {e(hu('aorta_hu'))}; portal vein {e(hu('portal_vein_hu'))}. {e(ph.get('note') or '')}" if ph.get("estimate") else "No phase estimate was stored for this scan.", base)]
    m = rec["model"]; a = rec["agent"]; st += [Paragraph("Model, settings and software", h), Paragraph(f"{e(m['name'])}, network {e(m['network'])}. Checkpoint SHA-256 <font size=7>{e(m['checkpoint_sha256'])}</font>. Licence: {e(m['license'])}.", base),
           Paragraph(f"Default threshold: logit {rec['thresholds']['default_logit']:g} (probability {rec['thresholds']['default_probability']}); recalibrated: logit {rec['thresholds']['recalibrated_logit']:g} (probability {rec['thresholds']['recalibrated_probability']}). Software commit {e(str(rec['software']['glassbox_commit']))}.", base),
           Paragraph("Agent report", h), Paragraph(f"Written by {e(str(a['mode']))} ({e(str(a.get('provider_host')))}; models {e(json.dumps(a.get('models')))}); number check {'passed' if a.get('number_check_ok') else 'NOT passed'}.", small)]
    st += [Paragraph(e(p), base) for p in (a.get("report") or "").split("\n\n")]
    log = [["#", "Tool", "OK", "Arguments"]] + [[str(x["step"]), x["tool"], "yes" if x["ok"] else "NO", e(json.dumps(x.get("args")))] for x in rec["tool_call_log"]]
    lt = Table(log, colWidths=[8 * mm, 42 * mm, 12 * mm, 100 * mm]); lt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb"))]))
    st += [Paragraph("Tool-call log", h), lt, Paragraph("Limitations", h)] + [Paragraph(f"&bull; {e(x)}", base) for x in rec["limitations"] if x]
    st += [Spacer(1, 6), Paragraph(f"Integrity: SHA-256 of the JSON record <font size=7>{e(rec['integrity']['sha256'])}</font>. The JSON version carries the full tool results and the checksums of every input file.", small)]
    doc.build(st); return buf.getvalue()
