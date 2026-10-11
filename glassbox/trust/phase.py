"""Contrast-phase heuristic (RESEARCH HEURISTIC, not a validated classifier), exactly as pre-registered in results/trust/PREREG.md before it was checked on any HCC data.
Median HU inside the model's own aorta (class 6) and portal-and-splenic-vein (class 17) masks, each eroded by one voxel; the aorta is limited to the slab 20 mm below to 20 mm
above the predicted liver; a vessel needs >= 200 voxels at 1.5 mm (675 mm3) to count. Thresholds come from the literature (late-arterial aortic enhancement about 245-300 HU;
unenhanced blood about 35-50 HU), not from the HCC data. DICOM tags, when present, are reported next to the estimate and never override it."""
from __future__ import annotations

import re

import numpy as np

from glassbox.agent.tools import qty

UNENHANCED_BELOW, ARTERIAL_FROM, PV_ENHANCED = 100.0, 200.0, 100.0   # HU
MIN_VOLUME_MM3 = 200 * 1.5 ** 3
SLAB_MM = 20.0
STATUS = "unvalidated heuristic (agreement with reference contrast sign at chance level, kappa 0.04)"  # results/SUMMARY.md section 7; the estimate is kept in the audit record only
RULE = "research heuristic v1 (thresholds from the literature, fixed before checking on HCC-TACE-Seg; results/trust/PREREG.md)"
FAMILY = {"arterial": "arterial-like", "venous": "venous-like", "delayed": "venous-like", "unenhanced": "unenhanced"}


def _median_hu(ct: np.ndarray, mask: np.ndarray, zooms, z_range: tuple[int, int] | None) -> tuple[float | None, int]:
    from scipy import ndimage   # imported here: the serve image has no scipy and only needs STATUS
    m = ndimage.binary_erosion(mask.astype(bool), iterations=1)
    if z_range is not None:
        keep = np.zeros_like(m); keep[:, :, z_range[0]:z_range[1] + 1] = True; m &= keep
    n = int(m.sum())
    return (float(np.median(ct[m])) if n * float(np.prod(zooms[:3])) >= MIN_VOLUME_MM3 else None), n


def tag_hint(tags: dict | None) -> dict:
    """What the DICOM header says about phase, if anything. `tags` maps DICOM keyword -> value (strings)."""
    tags = {k: str(v) for k, v in (tags or {}).items() if v not in (None, "")}
    desc = (tags.get("SeriesDescription", "") + " " + tags.get("ProtocolName", "")).lower(); hint = None
    if re.search(r"non[- ]?contrast|pre[- ]?contrast|unenhanced|\bplain\b|\bnative\b", desc):
        hint = "unenhanced"
    elif re.search(r"\barterial\b|\bart\b|\bhap\b|\blate art", desc):
        hint = "arterial"
    elif re.search(r"portal|venous|\bpvp?\b", desc):
        hint = "venous"
    elif re.search(r"delay|equilib", desc):
        hint = "delayed"
    return {"present": tags, "hint": hint}


def estimate_phase(ct: np.ndarray, aorta: np.ndarray, portal_vein: np.ndarray, liver: np.ndarray, zooms, tags: dict | None = None) -> dict:
    """ct in HU and masks in the canonical orientation (axis 2 = axial). Returns the contents of phase.json."""
    zs = np.flatnonzero(liver.any(axis=(0, 1))); pad = int(round(SLAB_MM / zooms[2]))
    z_range = (max(int(zs.min()) - pad, 0), min(int(zs.max()) + pad, ct.shape[2] - 1)) if len(zs) else None
    a, na = _median_hu(ct, aorta, zooms, z_range); p, npv = _median_hu(ct, portal_vein, zooms, None)
    if a is None:
        est = "indeterminate"
    elif a < UNENHANCED_BELOW:
        est = "unenhanced"
    elif a >= ARTERIAL_FROM:
        est = "arterial-like"
    elif p is not None and p >= PV_ENHANCED:
        est = "venous-like"
    else:
        est = "indeterminate"
    th = tag_hint(tags); fam = FAMILY.get(th["hint"]) if th["hint"] else None
    out = {"estimate": est, "aorta_hu": qty(round(a, 1), "HU") if a is not None else None, "portal_vein_hu": qty(round(p, 1), "HU") if p is not None else None,
           "aorta_voxels": na, "portal_vein_voxels": npv, "tags_present": th["present"], "tag_hint": th["hint"], "tag_disagrees": bool(fam and est != "indeterminate" and fam != est), "rule": RULE}
    out["note"] = ("The DICOM header and the HU estimate disagree: check the phase." if out["tag_disagrees"] else "No DICOM phase information was available; the estimate uses HU only." if not th["present"] else "The DICOM header agrees with, or does not contradict, the HU estimate.")
    return out
