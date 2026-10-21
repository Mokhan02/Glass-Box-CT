"""Re-screen of all 89 clean HCC cases with the Dice-maximising translation test (glassbox.seg.w3_lib.shift_test) over +-40 VOXELS in every axis (through-plane included), with BOTH model A (swin, shipped) and model D (nnU-Net default) predicted-liver masks; no tumour metric.
Flag: a model's best shift > 5 mm on some axis AND organ Dice gain > 0.05. Reports where the two models agree. GT root: corrected conversion for HCC_019, the clean conversion for the 14 position-matched cases, else the original conversion. CPU only.
Usage (cwd = repo checkout): python scripts/dq_rescreen_translation.py <A masks dir for the 75> <D pred dir for the 75> <A masks dir for the 14> <D pred dir for the 14> <clean89 list> <out json>"""
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402

a75, d75, a14, d14, lst, out = sys.argv[1:7]; uid = {k[0] for k in L.lesion_table("hcc")}; cases = [l.strip() for l in open(lst)]; R = []; CLEAN = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii_clean/hcc_tace_seg"
for c in cases:
    is_uid = c in uid; root = L.case_root("hcc", c, L.HCC_ROOT) if is_uid else CLEAN; _, geo = L.lesion_rows("hcc", c, root, {}, {}) if is_uid else (None, __import__("glassbox.seg.w2_lib", fromlist=["x"]).case_geometry(root, c))
    if is_uid:
        seg, ann, sp = geo["seg"], geo["ann"], geo["spacing"]
    else:
        _, sp, seg, ann, *_ = geo
    A = np.load(os.path.join(a75 if is_uid else a14, "masks", f"swin__{c}.npz"))["organ_full"]; D = np.asarray(nib.load(os.path.join(d75 if is_uid else d14, f"{c}.nii.gz")).dataobj) >= 1
    rec = {"case": c, "route": "uid" if is_uid else "resolved"}
    for n, o in (("A", A), ("D", D)):
        st = L.shift_test(o, seg, ann, sp, search_vox=(40, 40, 40)); rec[n] = {k: st[k] for k in ("dice_unshifted", "best_dice", "gain", "best_shift_mm", "max_axis_shift_mm")}; rec[n]["flag"] = bool(st["max_axis_shift_mm"] > L.SHIFT_MAX_MM and st["gain"] > L.SHIFT_MIN_GAIN)
    rec["agree"] = rec["A"]["flag"] == rec["D"]["flag"]; R.append(rec); print(c, "A", rec["A"]["best_shift_mm"], rec["A"]["gain"], rec["A"]["flag"], "| D", rec["D"]["best_shift_mm"], rec["D"]["gain"], rec["D"]["flag"], flush=True)
fa, fd = [r["case"] for r in R if r["A"]["flag"]], [r["case"] for r in R if r["D"]["flag"]]
json.dump({"search": "+-40 voxels per axis", "flag_rule": "best shift > 5 mm on some axis AND organ Dice gain > 0.05, either model", "n_cases": len(R), "flag_A": fa, "flag_D": fd, "flag_both": sorted(set(fa) & set(fd)), "flag_either": sorted(set(fa) | set(fd)), "only_A": sorted(set(fa) - set(fd)), "only_D": sorted(set(fd) - set(fa)), "per_case": R}, open(out, "w"), indent=1)
print(f"DONE {len(R)} cases | A flags {fa} | D flags {fd} | both {sorted(set(fa) & set(fd))}")
