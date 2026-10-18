"""W2-M M3 runner (results/w2_mech/PREREG.md): kidney contrast counterfactual. For every lesion in results/w2_mech/m3_arms.csv: the unedited run (with token recording, for the exploratory projection), the edit run and the sign-keeping control run, through the unchanged preprocessing
and the kidney arm (label 133 + point label 1 at the lesion's interior point). Edit = glassbox.bodycomp.counterfactual.edit_lesion (constant HU offset inside the GT mask, 1-voxel feather, clipped; realised mean recorded, flagged if > 5 HU off).
Exploratory projection: mean score of the kidney-native probe of the case's fold (M1 branch = ORGAN-SPECIFIC) over the mid tokens whose centre lies inside the lesion (original-grid voxel of the token centre). The first 10 unedited runs must reproduce the W2 v3 CSV (dice_full, coverage within 2e-5); exit 1 otherwise.
All predicted masks saved. Usage: python scripts/w2m_m3_run.py <inferer_dir> <weights> <erasers.npz> <outdir> <scratch_dir>"""
import csv
import os
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from glassbox.bodycomp.counterfactual import edit_lesion  # noqa: E402
from glassbox.seg.w2_lib import Runner, case_geometry, interior_point, lesion_metrics  # noqa: E402

inferer_dir, weights, erasers, outdir, scratch = sys.argv[1:6]
KITS = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii/c4kc_kits"
E = np.load(erasers); os.makedirs(os.path.join(outdir, "masks"), exist_ok=True); os.makedirs(scratch, exist_ok=True)
arms = list(csv.DictReader(open("results/w2_mech/m3_arms.csv")))
pilot = {(r["case"], int(r["rank"])) for r in csv.DictReader(open("results/w2_kidney/v2_pilot_selection.csv")) if r["organ"] == "kidney"}
table = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/gt_lesions.csv")) if int(r["rank"]) <= 2}
cases_all = sorted({c for c, r in table if (c, r) not in pilot}); case_idx = {c: i for i, c in enumerate(cases_all)}
v3 = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/v3/w2v2_main_kidney.csv")) if r["arm"] == "kidney_133_1"}
R = Runner(inferer_dir, weights); rows = []; t0 = time.time(); checked = 0; ok = True
by_case = {}
for a in arms:
    by_case.setdefault(a["case"], []).append(a)


def projection(case, M, comp, tokens):
    if not tokens:
        return float("nan"), 0
    f = case_idx[case] % 5; mean, scale, coef, b = E[f"probe_f{f}_mean"], E[f"probe_f{f}_scale"], E[f"probe_f{f}_coef"], float(E[f"probe_f{f}_intercept"][0])
    F = np.concatenate([t[0] for t in tokens]); xyz = np.concatenate([t[1] for t in tokens]).astype(float)
    vox = np.rint((np.linalg.inv(M) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]).astype(int); inb = np.all((vox >= 0) & (vox < np.array(comp.shape)), axis=1)
    inside = np.zeros(len(vox), bool); inside[inb] = comp[vox[inb, 0], vox[inb, 1], vox[inb, 2]]
    if not inside.any():
        return float("nan"), 0
    return float((((F[inside] - mean) / scale) @ coef + b).mean()), int(inside.sum())


for ci, (case, lesions) in enumerate(sorted(by_case.items())):
    nii, sp, seg, ann, zidx, lab, sel = case_geometry(KITS, case); ct = np.asarray(nii.dataobj).astype(np.float32); src = os.path.join(KITS, case, "ct.nii.gz")
    for a in sorted(lesions, key=lambda r: int(r["rank"])):
        rank = int(a["rank"]); diam, cid = sel[rank - 1]; assert abs(float(a["diam_mm"]) - diam) < 0.01
        comp_c = lab == cid; pt_c = interior_point(comp_c, sp); comp = np.zeros(seg.shape, bool); comp[:, :, ann] = comp_c; pt = (pt_c[0], pt_c[1], int(zidx[pt_c[2]]))
        for cond in ("baseline", "edit", "control"):
            info = {"target_mean": float("nan"), "realised_mean": float("nan"), "flagged": False}
            if cond == "baseline":
                path = src
            else:
                tgt = float(a["edit_target"] if cond == "edit" else a["control_target"]); edited, inf = edit_lesion(ct, comp, tgt); assert np.array_equal(edited[~comp], ct[~comp]), "edit changed voxels outside the mask"
                path = os.path.join(scratch, f"{case}_r{rank}_{cond}.nii"); nib.save(nib.Nifti1Image(edited, nii.affine, nii.header), path); info = inf
            try:
                data, img, M = R.prep(path); R.record(True); R.set_eraser(None)
                pred = R.predict(data, img, M, 133, 1, pt)[: seg.shape[0], : seg.shape[1], : seg.shape[2]]; toks = R.tokens; R.record(False)
            finally:
                if cond != "baseline" and os.path.exists(path):
                    os.remove(path)
            m = lesion_metrics(pred[:, :, ann], comp_c, sp); proj, ntok = projection(case, M, comp, toks)
            if cond == "baseline" and checked < 10:
                ref = v3[(case, rank)]; good = abs(m["coverage"] - float(ref["coverage"])) <= 2e-5 and abs(m["dice_full"] - float(ref["dice_full"])) <= 2e-5; ok &= good; checked += 1
                print(f"baseline check {case} r{rank}: {'MATCH' if good else 'MISMATCH'}", flush=True)
                if not ok:
                    print("BASELINE DETERMINISM CHECK FAILED"); sys.exit(1)
            rows.append({"arm": a["arm"], "case": case, "rank": rank, "condition": cond, "diam_mm": a["diam_mm"], "contrast_hu": a["contrast_hu"], "T": a["T"], "L": a["L"], "target_mean": info["target_mean"], "realised_mean": info["realised_mean"], "flagged": info["flagged"],
                         **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}, "n_lesion_tokens": ntok, "projection_native_probe": proj})
            np.savez_compressed(os.path.join(outdir, "masks", f"{cond}__{case}__r{rank}.npz"), pred=pred, point=np.array(pt))
    if ci % 10 == 0:
        print(f"[m3] {ci + 1}/{len(by_case)} cases, {len(rows)} runs, elapsed {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(outdir, "w2m_m3_runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(outdir, "w2m_m3_runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"DONE m3: {len(rows)} runs, {time.time() - t0:.0f}s", flush=True)
