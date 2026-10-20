"""Amendment 2 alignment gate as stated by the researcher: Dice-maximising translation test (<= 5 mm per axis) of the predicted organ against the GT organ, plus the shape and axis-flip checks, for every case (glassbox.seg.w3_lib.shift_test, alignment_report).
Case verdicts: MISALIGNED (best shift > 5 mm on some axis AND it raises organ Dice by >= 0.05), MODEL-SIDE (shift > 5 mm but no translation substantially fixes it), ALIGNED. The gate stops the run if more than 2 cases are MISALIGNED or fail shape / flip.
CPU only. Usage (cwd = repo checkout): python scripts/w3_shift_gate.py A <model A masks dir (w3_modelA_run.py output, masks/*.npz with organ_full)> <out json>
                                                                       python scripts/w3_shift_gate.py D <nnU-Net prediction dir> <out json>"""
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402

mode, src, out = sys.argv[1:4]; table = L.lesion_table("hcc"); cases = sorted({k[0] for k in table}); R = []
for c in cases:
    root = L.case_root("hcc", c, L.HCC_ROOT); _, geo = L.lesion_rows("hcc", c, root, {}, {}); seg, ann, sp = geo["seg"], geo["ann"], geo["spacing"]
    if mode == "A":
        organ = np.load(os.path.join(src, "masks", f"swin__{c}.npz"))["organ_full"]
    else:
        organ = np.asarray(nib.load(os.path.join(src, f"{c}.nii.gz")).dataobj) >= 1
    rep = L.alignment_report(organ, seg, ann, sp, nib.load(os.path.join(root, c, "ct.nii.gz")).shape); st = L.shift_test(organ, seg, ann, sp)
    rec = {"case": c, "organ_dice": rep["organ_dice"], "centroid_offset_mm": rep["centroid_offset_mm"], "shape_ok": rep["shape_ok"], "flip_improves_dice": rep["flip_improves_dice"], **{f"shift_{k}": v for k, v in st.items()}}
    rec["pass"] = bool(rep["shape_ok"] and not rep["flip_improves_dice"] and st["verdict"] != "MISALIGNED"); R.append(rec); print(c, rec["organ_dice"], "centroid", rec["centroid_offset_mm"], "shift", st["best_shift_mm"], "gain", st["gain"], st["verdict"], flush=True)
fails = [r["case"] for r in R if not r["pass"]]
json.dump({"model": mode, "gate": "shape + no flip improves Dice + shift test (best Dice-maximising translation <= 5 mm per axis, else MISALIGNED only if gain >= 0.05)", "n_cases": len(R), "n_misaligned_or_failed": len(fails), "failures": fails,
           "model_side": [r["case"] for r in R if r["shift_verdict"] == "MODEL-SIDE"], "centroid_gt_15mm": [r["case"] for r in R if r["centroid_offset_mm"] > L.ALIGN_MAX_OFFSET_MM], "status": "STOP" if len(fails) > L.ALIGN_MAX_FAILURES else "OK", "per_case": R}, open(out, "w"), indent=1)
print(f"DONE {mode}: {len(R)} cases, MISALIGNED/failed {fails}, model-side {[r['case'] for r in R if r['shift_verdict'] == 'MODEL-SIDE']}, centroid > 15 mm {[r['case'] for r in R if r['centroid_offset_mm'] > L.ALIGN_MAX_OFFSET_MM]}")
