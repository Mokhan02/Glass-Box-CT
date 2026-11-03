"""W4 (WAW-TACE) scoring (CPU; results/w4_waw/PREREG.md): per-tumour detection of one model arm on every series in a ground-truth root (`glassbox/seg/w4_waw.py`: the W3 Stage 1 definition, coverage >= 0.10), plus the alignment gate of the
PREREG (the model's liver prediction against the TotalSegmentator-derived organ of the ground truth, the W3 Amendment 2 checks: shape equals the input, no axis flip raises the organ Dice, centroid offset <= 15 mm, and the Dice-maximising
translation <= 5 mm per axis). Predictions: `<pred dir>/<case>.npz` with bool arrays `tumour` (and `liver`), or the nnU-Net label map `<case>.nii.gz` (tumour = label 2, liver = label >= 1).
Usage (python with nibabel and scipy): python scripts/w4_waw_score.py <gt root> <pred dir> <arm> <out dir> [--workers N] [--cases 14_1 14_2 ...]   (N > 1: series scored in N forked processes, same files as N = 1)"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg import w4_waw as W  # noqa: E402


def load_pred(pred_dir, case):
    p = os.path.join(pred_dir, case + ".npz")
    if os.path.exists(p):
        z = np.load(p); return z["tumour"].astype(bool), (z["liver"].astype(bool) if "liver" in z.files else None)
    import nibabel as nib
    a = np.asarray(nib.load(os.path.join(pred_dir, case + ".nii.gz")).dataobj); return a == 2, a >= 1


_CTX = {}


def score_case(case):
    """Rows and the alignment-gate entry (or None) of one series; a pure function of the series, so the output does not depend on the number of workers."""
    gt, pred_dir, arm = _CTX["gt"], _CTX["pred_dir"], _CTX["arm"]
    ser = W.read_series(gt, case); tumour, liver = load_pred(pred_dir, case); rows = W.score_arrays(arm, ser["seg"], ser["labels"], ser["labels_dil2"], ser["tumours"], tumour, ser["spacing"]); al = None
    if liver is not None:
        ann = (ser["seg"] > 0).any(axis=(0, 1)); rep = L.alignment_report(liver, ser["seg"], ann, ser["spacing"], ser["ct_shape"]); sh = L.shift_test(liver, ser["seg"], ann, ser["spacing"])
        al = {"case": case, **rep, "shift_verdict": sh["verdict"], "best_shift_mm": sh["best_shift_mm"], "gate_pass": bool(rep["pass"] and sh["verdict"] in ("ALIGNED", "MODEL-SIDE")), "n_tumours": len(ser["tumours"])}
    return rows, al


def main(argv):
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 1
    gt, pred_dir, arm, out = argv[:4]; cases = argv[argv.index("--cases") + 1:] if "--cases" in argv else sorted(d for d in os.listdir(gt) if os.path.isdir(os.path.join(gt, d)) and d != "__pycache__")
    os.makedirs(out, exist_ok=True); rows, align = [], []; _CTX.update(gt=gt, pred_dir=pred_dir, arm=arm)
    if workers > 1:      # fork workers; imap keeps the order of `cases`, so the files are byte-identical to the serial run
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = list(pool.imap(score_case, cases, chunksize=1))
    else:
        res = [score_case(c) for c in cases]
    for r, al in res:
        rows += r
        if al is not None:
            align.append(al)
    with open(os.path.join(out, f"scores_{arm}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=W.FIELDS); w.writeheader(); w.writerows(rows)
    json.dump({"arm": arm, "series": len(cases), "series_gate_fail": [a["case"] for a in align if not a["gate_pass"]], "gate": "shape equals input, no flip improves, centroid offset <= 15 mm (W3 Amendment 2) and no MISALIGNED translation (> 5 mm per axis with a Dice gain >= 0.05)", "per_series": align},
              open(os.path.join(out, f"alignment_{arm}.json"), "w"), indent=1)
    print(f"{arm}: {len(cases)} series, {len(rows)} tumour rows, {sum(1 for a in align if not a['gate_pass'])} series failing the alignment gate")
    return rows, align


if __name__ == "__main__":
    main(sys.argv[1:])
