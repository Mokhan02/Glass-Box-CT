"""W2 v2 arms and pilot-gate logic (results/w2_kidney/PREREG_v2.md). Kept free of torch so it can be tested anywhere.
Point label convention (MONAI VISTA-3D): 1 = positive, 0 = negative for regular and zero-shot classes; 3 = positive, 2 = negative for the SPECIAL classes (tumour, vessel).
The 1->3 conversion is applied only in training/validation sampling, NOT to user points at inference, so the caller must send the right label."""
from __future__ import annotations

SPECIAL_INDEX = (23, 24, 25, 26, 27, 57, 128)   # monai/apps/vista3d/sampler.py
ZERO_SHOT_MIN = 133                              # the model treats class indices above 132 as zero-shot

ARMS = {
    "kidney_133_1": {"organ": "kidney", "label": 133, "point_label": 1, "role": "primary"},
    "hcc_26_3": {"organ": "hcc", "label": 26, "point_label": 3, "role": "primary"},
    "hcc_133_1": {"organ": "hcc", "label": 133, "point_label": 1, "role": "primary"},
    "hcc_26_1": {"organ": "hcc", "label": 26, "point_label": 1, "role": "secondary_v1_equivalent"},
}


def expected_point_label(label: int) -> int:
    """The positive point label the model expects for a class prompt."""
    return 3 if label in SPECIAL_INDEX else 1


def lesion_passes(vol_ratio: float, inside_share: float) -> bool:
    return vol_ratio < 3.0 and inside_share > 0.3


def arm_passes(rows: list[dict]) -> dict:
    """rows: dicts with pred_over_lesion_volume, pred_frac_in_lesion, stratum. Arm passes if >= 80% of lesions pass and every stratum present passes >= 60%."""
    ok = [lesion_passes(r["pred_over_lesion_volume"], r["pred_frac_in_lesion"]) for r in rows]
    strata = {}
    for r, o in zip(rows, ok):
        strata.setdefault(r["stratum"], []).append(o)
    frac = sum(ok) / len(ok); sfrac = {k: sum(v) / len(v) for k, v in strata.items()}
    return {"n": len(ok), "pass_fraction": frac, "strata": sfrac, "passes": bool(frac >= 0.8 and all(v >= 0.6 for v in sfrac.values()))}
