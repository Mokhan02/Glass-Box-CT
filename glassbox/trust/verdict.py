"""Trusted / Review / Unreliable, for a whole scan and for each lesion, with plain reasons that each name the tool result they come from.
The rules are a transparent product heuristic, written down here and not tuned on the demo scans: they combine the quality flag (the only validated signal, AUROC 0.925 in
cross-validation), the known bright-tumour blind spot (results/SUMMARY.md), the contrast-phase heuristic (results/trust/PREREG.md), and simple size and fragment rules.
This is a statement about how far a MEASUREMENT can be relied on. It is never a diagnosis."""
from __future__ import annotations

TRUSTED, REVIEW, UNRELIABLE = "Trusted", "Review", "Unreliable"
ORDER = {TRUSTED: 0, REVIEW: 1, UNRELIABLE: 2}
FLAG_NEAR_BAND = 0.25      # a flag probability from here up to the flag threshold is "close to flagged": Review
FRAGMENT_MM, FRAGMENT_CM3 = 3.0, 0.1   # as the lesions table groups them
MEASURABLE_MM = 10.0       # RECIST 1.1 measurable size
LOW_CONFIDENCE = 0.6       # mean model probability inside a lesion
MANY_FRAGMENTS = 5

PLAIN = {
    "predicted tumour volume": "the size of the predicted tumour",
    "mean model probability inside the predicted tumour": "how confident the model is inside its predicted tumour",
    "mean prediction entropy inside the predicted tumour": "how uncertain the model's voxel decisions are inside it",
}


def _r(code: str, text: str, source: str, level: str) -> dict:
    return {"code": code, "text": text, "source": source, "level": level}


def is_fragment(l: dict) -> bool:
    return l["longest_axial_diameter"]["value"] < FRAGMENT_MM or l["volume"]["value"] < FRAGMENT_CM3


def blind_spot(lesions: dict, phase: dict | None) -> list[dict]:
    """The situation in which this model tends to miss tumours: tumour brighter than the surrounding liver, or an arterial-like phase (research heuristic)."""
    out = []
    c = (lesions or {}).get("tumour_liver_contrast")
    if c and c["value"] > 0:
        out.append(_r("bright_tumour", f"The predicted tumour is brighter than the surrounding liver ({c['value']} HU). This model is known to miss some liver tumours in that situation (research finding).", "measure_lesions", REVIEW))
    if phase and phase.get("estimate") == "arterial-like":
        out.append(_r("arterial_phase", f"The scan looks arterial-phase by a research heuristic (aorta {phase['aorta_hu']['value']} HU); liver tumours are often brighter than liver in that phase.", "phase_check", REVIEW))
    return out


def assess(lesions: dict | None, quality: dict | None, phase: dict | None = None) -> dict:
    """lesions: lesions.json, quality: the tumour entry of quality_audit.json['structures'], phase: phase.json or None. Returns the scan verdict and one per lesion."""
    lesions = lesions or {}; ls = lesions.get("lesions") or []; reasons: list[dict] = []; level = TRUSTED
    bad = blind_spot(lesions, phase)
    if quality is None or not quality.get("assessed"):
        level = REVIEW; reasons.append(_r("no_tumour", "No tumour was predicted. That does not mean there is none: on held-out data about one lesion in five was largely missed.", "quality_audit", REVIEW))
    else:
        p, thr = quality["failure_probability"], quality.get("flag_threshold_probability", 0.46)
        drivers = [PLAIN.get(d["name"], d["name"]) for d in (quality.get("drivers") or []) if d["pushes_toward"] == ("failure" if quality["flag"] == "likely wrong" else "success")][:2]
        why = (" because " + " and ".join(drivers) + " look like cases where it got the tumour " + ("wrong" if quality["flag"] == "likely wrong" else "right")) if drivers else ""
        if quality["flag"] == "likely wrong":
            level = UNRELIABLE; reasons.append(_r("flag", f"The quality flag estimates a {p} probability that this tumour segmentation is wrong (flagged at {thr:.2f}){why}.", "quality_audit", UNRELIABLE))
        elif p >= FLAG_NEAR_BAND:
            level = REVIEW; reasons.append(_r("flag_near", f"The quality flag is not triggered but is close (probability {p}, flagged at {thr:.2f}).", "quality_audit", REVIEW))
        else:
            reasons.append(_r("flag_ok", f"The quality flag is quiet (probability {p} that the segmentation is wrong){why}.", "quality_audit", TRUSTED))
    if phase and phase.get("estimate") == "unenhanced":
        level = max(level, REVIEW, key=ORDER.get); reasons.append(_r("unenhanced", f"The scan looks unenhanced (aorta {phase['aorta_hu']['value']} HU); this model's tumour measurements were only examined on contrast-enhanced scans.", "phase_check", REVIEW))
    for b in bad:
        level = max(level, REVIEW, key=ORDER.get); reasons.append(b)
    nfrag = sum(is_fragment(l) for l in ls)
    if nfrag >= MANY_FRAGMENTS:
        level = max(level, REVIEW, key=ORDER.get); reasons.append(_r("fragments", f"{nfrag} very small fragments were predicted (under {FRAGMENT_MM:g} mm or {FRAGMENT_CM3:g} cm3); they are probably noise and should not be counted as lesions.", "measure_lesions", REVIEW))
    per = []
    for l in ls:
        lv, rs = TRUSTED, []
        d, v, mp = l["longest_axial_diameter"]["value"], l["volume"]["value"], l.get("mean_probability")
        if is_fragment(l):
            lv = UNRELIABLE; rs.append(_r("fragment", f"A fragment ({d} mm, {v} cm3), probably noise.", "measure_lesions", UNRELIABLE))
        else:
            if level == UNRELIABLE:
                lv = UNRELIABLE; rs.append(_r("scan_flag", "The scan's quality flag says this segmentation is likely wrong.", "quality_audit", UNRELIABLE))
            if d < MEASURABLE_MM:
                lv = max(lv, REVIEW, key=ORDER.get); rs.append(_r("small", f"Under the {MEASURABLE_MM:g} mm RECIST measurable size ({d} mm).", "measure_lesions", REVIEW))
            if mp is not None and mp < LOW_CONFIDENCE:
                lv = max(lv, REVIEW, key=ORDER.get); rs.append(_r("low_confidence", f"Low model confidence inside this lesion (mean probability {mp}).", "measure_lesions", REVIEW))
            if bad and lv == TRUSTED:
                lv = REVIEW; rs.append(bad[0])
        if not rs:
            rs.append(_r("ok", "No warning signs from the flag, size, confidence or contrast.", "measure_lesions", TRUSTED))
        per.append({"id": l["id"], "level": lv, "reasons": rs})
    return {"level": level, "reasons": reasons, "lesions": per, "rules": "research heuristic: transparent rules in glassbox/trust/verdict.py; not tuned on the demo scans; not a diagnosis"}


def sentence(level: str) -> str:
    return {TRUSTED: "This measurement looks reliable for this case", REVIEW: "Review this measurement before relying on it", UNRELIABLE: "Don't rely on this measurement"}[level]
