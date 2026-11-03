"""W4 (WAW-TACE): comparison of an instance's ground-truth build (scripts/w4_waw_paired_contrast.py --save-dir, half of the patients) against the cohort run on host1 (results/w4_waw/paired_host1). The registration is multithreaded and not bit-reproducible
(DEVIATIONS 18), so the check FAILS (exit 1) only on a structural change: a tumour missing on either side, a changed exclusion category, or a changed patient QA pass/fail. Every other difference is written to <out>/gt_vs_host1.json (distribution of the
contrast differences, sign changes, transform changes, largest differences, QA metric differences) and <out>/gt_vs_host1_differences.csv (all contrast differences > 0.01 HU), not judged.
Usage: python scripts/w4_waw_gt_compare.py <host1 paired dir> <instance paired dir> <out dir>"""
import csv
import json
import os
import sys

import numpy as np

COLS = ["c_drawn", "c_other", "c_arterial", "c_portal", "c_arterial_eroded", "c_portal_eroded", "c_arterial_median", "c_portal_median"]
QCOLS = ["liver_dice_final", "propagated_inside_target_liver", "volume_ratio", "tumour_centroid_shift_mm"]


def flt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(d):
    return ({(r["patient"], r["rank"]): r for r in csv.DictReader(open(os.path.join(d, "tumours_paired.csv")))}, {r["patient"]: r for r in csv.DictReader(open(os.path.join(d, "patient_qa.csv")))})


def compare(ref_dir, new_dir, out):
    tr, qr = load(ref_dir); tn, qn = load(new_dir); os.makedirs(out, exist_ok=True); pats = set(qn)
    res = {"patients": len(pats), "tumours_new": len(tn), "tumours_ref_of_these_patients": sum(1 for k in tr if k[0] in pats)}
    res["tumours_missing_in_ref"] = sorted(f"{k[0]}:{k[1]}" for k in tn if k not in tr); res["tumours_missing_in_new"] = sorted(f"{k[0]}:{k[1]}" for k in tr if k[0] in pats and k not in tn)
    both = [k for k in tn if k in tr]
    res["category_changes"] = [{"tumour": f"{k[0]}:{k[1]}", "ref": tr[k]["category"], "new": tn[k]["category"]} for k in both if tr[k]["category"] != tn[k]["category"]]
    res["qa_pass_changes"] = [p for p in pats if p in qr and qn[p]["qa_pass"] != qr[p]["qa_pass"]]; res["transform_changes"] = [{"patient": p, "ref": qr[p]["transform"], "new": qn[p]["transform"]} for p in pats if p in qr and qr[p]["transform"] != qn[p]["transform"]]
    res["sign_changes"] = [{"tumour": f"{k[0]}:{k[1]}", "column": c, "ref": tr[k][c], "new": tn[k][c]} for k in both for c in ("c_arterial", "c_portal") if flt(tr[k][c]) is not None and flt(tn[k][c]) is not None and (flt(tr[k][c]) > 0) != (flt(tn[k][c]) > 0)]
    res["contrast_difference_HU"] = {}; rows = []
    for c in COLS:
        d = [(abs(flt(tr[k][c]) - flt(tn[k][c])), k) for k in both if flt(tr[k][c]) is not None and flt(tn[k][c]) is not None]; a = np.array([x for x, _ in d])
        if len(a):
            res["contrast_difference_HU"][c] = {"n": int(len(a)), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "n_gt_0.01": int((a > 0.01).sum()), "n_gt_1": int((a > 1).sum())}
        rows += [{"tumour": f"{k[0]}:{k[1]}", "column": c, "ref": tr[k][c], "new": tn[k][c], "abs_diff": round(x, 3), "transform_ref": qr[k[0]]["transform"], "transform_new": qn[k[0]]["transform"]} for x, k in d if x > 0.01]
    res["largest_differences"] = sorted(rows, key=lambda r: -r["abs_diff"])[:5]
    res["qa_metric_max_abs_difference"] = {c: max((abs(flt(qr[p][c]) - flt(qn[p][c])) for p in pats if p in qr and flt(qr[p][c]) is not None and flt(qn[p][c]) is not None), default=None) for c in QCOLS}
    res["structural_ok"] = not (res["tumours_missing_in_ref"] or res["tumours_missing_in_new"] or res["category_changes"] or res["qa_pass_changes"])
    json.dump(res, open(os.path.join(out, "gt_vs_host1.json"), "w"), indent=1)
    with open(os.path.join(out, "gt_vs_host1_differences.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tumour", "column", "ref", "new", "abs_diff", "transform_ref", "transform_new"]); w.writeheader(); w.writerows(sorted(rows, key=lambda r: -r["abs_diff"]))
    return res


if __name__ == "__main__":
    r = compare(*sys.argv[1:4]); print(json.dumps({k: v for k, v in r.items() if k != "largest_differences"}, indent=1)); sys.exit(0 if r["structural_ok"] else 1)
