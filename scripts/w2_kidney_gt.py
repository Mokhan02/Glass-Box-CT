"""W2 Gate 2: ground-truth-only feasibility for C4KC-KiTS (results/w2_kidney/PREREG.md). NO model is run.
Lesion = 26-connected component of Mass (label 2), longest axial Feret diameter >= 10 mm, two largest per case; contrast = lesion HU minus Kidney (label 1) parenchyma in the
5-20 mm shell (glassbox/audit/contrast.py, the same function as W1). In-distribution flag on every output: possibly seen in training (images, via AbdomenCT-1K).
Usage: python scripts/w2_kidney_gt.py <heldout_nii/c4kc_kits dir> results/w2_kidney"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from glassbox.audit.contrast import case_lesions  # noqa: E402

root, out = sys.argv[1:3]
FLAG = "possibly seen in training (images, via AbdomenCT-1K)"


def one(case):
    nii = nib.load(os.path.join(root, case, "ct.nii.gz")); sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
    ct = np.asarray(nii.dataobj).astype(np.float32); seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj)
    rows = case_lesions(ct, seg == 2, seg == 1, sp, n_lesions=None)
    return [{"case": case, "in_distribution_flag": FLAG, **r} for r in rows], {"case": case, "kidney_vox": int((seg == 1).sum()), "mass_vox": int((seg == 2).sum())}


if __name__ == "__main__":
    cases = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    with Pool(8) as pool:
        res = pool.map(one, cases)
    rows = [r for rs, _ in res for r in rs]; inv = [i for _, i in res]
    with open(os.path.join(out, "gt_lesions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    prim = [r for r in rows if r["rank"] <= 2 and not np.isnan(r["contrast_hu"])]
    by = {}
    for r in prim:
        by.setdefault(r["case"], []).append(r["contrast_hu"] > 0)
    cs = sorted(by); cnt = np.array([[sum(by[c]), len(by[c])] for c in cs]); rng = np.random.default_rng(0)
    idx = rng.integers(0, len(cs), (1000, len(cs))); draws = cnt[idx, 0].sum(1) / cnt[idx, 1].sum(1)
    c = np.array([r["contrast_hu"] for r in prim]); d = np.array([r["diam_mm"] for r in prim])
    res_json = {"in_distribution_flag": FLAG, "cases_total": len(cases), "cases_with_lesion_ge_10mm": len(cs), "cases_without": sorted(set(cases) - set(cs)),
                "lesions_all_ge_10mm": len(rows), "lesions_primary_two_largest": len(prim), "bright_lesions": int((c > 0).sum()), "dark_lesions": int((c < 0).sum()),
                "bright_share": float((c > 0).mean()), "bright_share_ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "bright_cases": int(sum(any(v) for v in by.values())), "fallback_lesions": int(sum(r["fallback"] for r in prim)),
                "contrast_quantiles_5_25_50_75_95": [float(np.percentile(c, q)) for q in (5, 25, 50, 75, 95)],
                "diameter_quantiles_mm_5_25_50_75_95": [float(np.percentile(d, q)) for q in (5, 25, 50, 75, 95)],
                "bright_by_size": {n: [int(((c > 0) & m).sum()), int(m.sum())] for n, m in (("<30 mm", d < 30), ("30-60 mm", (d >= 30) & (d <= 60)), (">60 mm", d > 60))},
                "underpowered_for_brightness_test_lt_40_bright": bool((c > 0).sum() < 40)}
    json.dump(res_json, open(os.path.join(out, "gt_summary.json"), "w"), indent=1)
    print(json.dumps(res_json, indent=1))
