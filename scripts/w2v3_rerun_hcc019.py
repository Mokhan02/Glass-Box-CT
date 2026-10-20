"""Re-run of the two W2 v3 HCC rows of HCC_019_773068 (arms hcc_26_3 and hcc_133_1, one positive point at the interior point of the lesion, as scripts/w2v2_run.py) with the CORRECTED label (HCC_019 correction, results/dq_hcc_labels/DQ_REPORT.md),
using the same Runner / prediction code as the W2-M runs (glassbox/seg/w2_lib.py). Determinism check first: the same code on the AS-CONVERTED label must reproduce the stored v3 rows (coverage and dice_full within 2e-5, same point). Writes the corrected rows in the v3 CSV format.
Usage (cwd = repo checkout, GPU): python scripts/w2v3_rerun_hcc019.py <vendored_inferer_dir> <weights> <stored v3 main hcc csv> <outdir>"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg.w2_arms import ARMS  # noqa: E402
from glassbox.seg.w2_lib import Runner, axial_diam, case_geometry, dice_within, interior_point, lesion_metrics  # noqa: E402
from scipy import ndimage  # noqa: E402

inferer_dir, weights, stored_csv, outdir = sys.argv[1:5]; os.makedirs(outdir, exist_ok=True); CASE = "HCC_019_773068"
stored = {(r["arm"], int(r["rank"])): r for r in csv.DictReader(open(stored_csv)) if r["case"] == CASE}
R = Runner(inferer_dir, weights); data, img, M = R.prep(os.path.join(L.HCC_ROOT, CASE, "ct.nii.gz")); arms = [a for a, c in ARMS.items() if c["organ"] == "hcc" and a in ("hcc_26_3", "hcc_133_1")]


def run(root, table, tag):
    nii, sp, seg, ann, zidx, lab, sel = case_geometry(root, CASE); out = []
    for rank, (diam, cid) in enumerate(sel[:2], 1):
        if (CASE, rank) not in table:
            continue
        ref = table[(CASE, rank)]; comp_c = lab == cid; pt_c = interior_point(comp_c, sp); pt = (pt_c[0], pt_c[1], int(zidx[pt_c[2]]))
        for arm in arms:
            cfg = ARMS[arm]; t1 = time.time(); pred = R.predict(data, img, M, cfg["label"], cfg["point_label"], pt)[:seg.shape[0], :seg.shape[1], :seg.shape[2]]; dt = time.time() - t1
            pa = pred[:, :, ann]; m = lesion_metrics(pa, comp_c, sp); pcdiam = 0.0
            if pa.any():
                plab, _ = ndimage.label(pa, structure=np.ones((3, 3, 3))); pz = int(np.searchsorted(zidx, pt[2])); pc = int(plab[pt[0], pt[1], pz]); pcdiam = axial_diam(plab == pc, sp[0], sp[1]) if pc > 0 else 0.0
            out.append({"stage": "main", "arm": arm, "organ": "hcc", "label_prompt": cfg["label"], "point_label": cfg["point_label"], "case": CASE, "rank": rank, "group": "bright" if float(ref["contrast_hu"]) > 0 else "dark", "stratum": L.stratum(diam),
                        "diam_mm": round(diam, 3), "contrast_hu": float(ref["contrast_hu"]), "coverage": round(m["coverage"], 5), "dice_full": round(m["dice_full"], 5), "dice_within_10mm": round(dice_within(pa, comp_c, sp), 5), "pred_diam_mm": round(m["pred_diam_mm"], 3),
                        "point_comp_diam_mm": round(pcdiam, 3), "pred_over_lesion_volume": round(m["pred_over_lesion_volume"], 4), "pred_frac_in_lesion": round(m["pred_frac_in_lesion"], 4), "point_x": pt[0], "point_y": pt[1], "point_z": pt[2], "t_inference_s": round(dt, 2)})
    return out


chk = run(L.HCC_ROOT, L.lesion_table("hcc", as_converted=True), "as_converted"); det = []
for r in chk:
    s = stored[(r["arm"], r["rank"])]; ok = abs(float(s["coverage"]) - r["coverage"]) <= 2e-5 and abs(float(s["dice_full"]) - r["dice_full"]) <= 2e-5 and (int(s["point_x"]), int(s["point_y"]), int(s["point_z"])) == (r["point_x"], r["point_y"], r["point_z"])
    det.append({"arm": r["arm"], "stored_coverage": float(s["coverage"]), "new_coverage": r["coverage"], "stored_dice": float(s["dice_full"]), "new_dice": r["dice_full"], "same_point": (int(s["point_x"]), int(s["point_y"]), int(s["point_z"])) == (r["point_x"], r["point_y"], r["point_z"]), "ok": bool(ok)})
json.dump(det, open(os.path.join(outdir, "determinism_check_as_converted.json"), "w"), indent=1); print("DETERMINISM", json.dumps(det), flush=True)
assert all(d["ok"] for d in det), "the rerun code does not reproduce the stored v3 rows; stop and report"
new = run(L.case_root("hcc", CASE, L.HCC_ROOT), L.lesion_table("hcc"), "corrected")
with open(os.path.join(outdir, "w2v3_hcc019_corrected_rows.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(new[0])); w.writeheader(); w.writerows(new)
print("RERUN DONE", json.dumps([{k: r[k] for k in ("arm", "group", "contrast_hu", "coverage", "dice_full", "point_x", "point_y", "point_z")} for r in new]), flush=True)
