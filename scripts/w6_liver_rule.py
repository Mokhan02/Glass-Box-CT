"""W6 liver-label rule (PREREG W6 section 4), evaluated on the BASELINE only, before any arm trains. In MSD Task03 the liver label (1) excludes the tumour voxels (2), so the mapping [[1,1],[2,26]] would teach the liver prompt to output zero inside tumours. Statistic (fixed now):
the pooled fraction of ground-truth tumour voxels (label 2) that the baseline's class-1 mask covers, over the nnU-Net fold-0 VALIDATION cases that contain tumour voxels (the W4 route: `Runner.prep` / `predict_auto(data, img, 1)`, on the original grid). **Rule: fraction >= 0.50 -> every arm uses the tumour-only mapping [[2,26]]; otherwise [[1,1],[2,26]].**
Usage (python of the new environment, GPU, cwd = repo checkout): python scripts/w6_liver_rule.py <vendored_inferer_dir> <weights> <dataset_dir> <datalist.json> <out.json>"""
import json
import os
import sys

import numpy as np

THRESHOLD = 0.50


def pooled_fraction(pairs):
    """pairs: iterable of (liver_pred bool array, tumour_gt bool array); returns (pooled fraction over cases with tumour voxels, per-case list)."""
    cov = tot = 0; per = []
    for liv, tum in pairs:
        n = int(np.asarray(tum).sum())
        if n == 0:
            continue
        c = int((np.asarray(liv, bool) & np.asarray(tum, bool)).sum()); cov += c; tot += n; per.append(c / n)
    return (cov / tot if tot else float("nan")), per


def decide(frac):
    return "[[2,26]]" if frac >= THRESHOLD else "[[1,1],[2,26]]"


def main(argv):
    inf, wts, ddir, dl, out = argv[:5]
    import nibabel as nib
    sys.path.insert(0, "."); from glassbox.seg.w2_lib import Runner
    rows = [r for r in json.load(open(dl))["training"] if r["fold"] == 0]; R = Runner(inf, wts); pairs = []; names = []
    for r in rows:
        lab = np.asarray(nib.load(os.path.join(ddir, r["label"])).dataobj) == 2
        if not lab.any():
            continue
        data, img, _ = R.prep(os.path.join(ddir, r["image"])); liv = np.asarray(R.predict_auto(data, img, 1)).astype(bool)[:lab.shape[0], :lab.shape[1], :lab.shape[2]]; pairs.append((liv, lab)); names.append(r["image"]); print(r["image"], float((liv & lab).sum() / lab.sum()), flush=True)
    frac, per = pooled_fraction(pairs); rep = {"pooled_fraction_of_tumour_voxels_inside_baseline_liver": frac, "n_cases_with_tumour": len(per), "per_case": dict(zip(names, per)), "threshold": THRESHOLD, "mapping_decision": decide(frac)}
    json.dump(rep, open(out, "w"), indent=1); print(json.dumps({k: rep[k] for k in ("pooled_fraction_of_tumour_voxels_inside_baseline_liver", "n_cases_with_tumour", "mapping_decision")}))


if __name__ == "__main__":
    main(sys.argv[1:])
