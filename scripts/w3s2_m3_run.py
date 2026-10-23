"""W3 Stage 2, M3 runner, GPU (PREREG_stage2.md; contrast_cf procedure; research use only for model A). Per case: an unedited baseline pass (lesion coverage / Dice recomputed here, not taken from Stage 1) decides the arms: Arm A = baseline detected (coverage >= 0.10) and c < 0 (edit to L + |c|, control T - 2|c|);
Arm B = baseline coverage < 0.10 and c > 0 (edit to L - |c|, control T + 2|c|). Each lesion is edited ALONE in its own copy of the volume (glassbox.bodycomp.counterfactual.edit_lesion: constant HU offset inside the GT mask, 0.5 weight on the 1-voxel inner border, clip [-1024, 3071]); edited volumes
go through the unchanged preprocessing and the model's primary pipeline and are deleted after use. Also records, for the exploratory projection, the mean 'mid' feature over the tokens inside the lesion for the baseline and the edited volume.
Outputs in <outdir>: m3_lesions.csv (one row per lesion and condition base / edit / control), tokens/<case>_r<rank>.npz, done_cases.txt (resumable).
Usage (cwd = repo checkout, GPU): python scripts/w3s2_m3_run.py A <outdir> <repo_A> <swin weights> [--limit N] [--cases c ...] [--shard i/n]
                                   python scripts/w3s2_m3_run.py D <outdir> <nnU-Net model folder> [--limit N] [--cases c ...] [--shard i/n]"""
import csv
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, ".")
from glassbox.audit.contrast import case_lesions  # noqa: E402
from glassbox.bodycomp.counterfactual import arm_targets, edit_lesion  # noqa: E402
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg import w3s2_engines as EN  # noqa: E402
from glassbox.seg.w2_lib import case_geometry  # noqa: E402

model, outdir = sys.argv[1:3]; rest = sys.argv[3:]; limit = int(rest[rest.index("--limit") + 1]) if "--limit" in rest else None; only = rest[rest.index("--cases") + 1:] if "--cases" in rest else None
shard = rest[rest.index("--shard") + 1] if "--shard" in rest else None; pos = [a for a in rest if not a.startswith("--") and a not in (only or []) and a != (str(limit) if limit else None) and a != shard]
table = L.lesion_table("hcc", include_excluded=True); cases = sorted({k[0] for k in table}); cases = only or cases[:limit]
if shard:
    si, sn = (int(v) for v in shard.split("/")); cases = cases[si::sn]
os.makedirs(os.path.join(outdir, "tokens"), exist_ok=True); dev = torch.device("cuda"); eng = EN.EngineA(pos[0], pos[1], outdir, dev) if model == "A" else EN.EngineD(pos[0], outdir, dev)
FIELDS = ["case", "rank", "arm", "condition", "T", "L", "c", "target", "realised_mean", "flagged", "coverage", "dice_full", "dice_within_10mm", "pred_diam_mm", "n_tokens"]
dn = os.path.join(outdir, "done_cases.txt"); done = set(l.strip() for l in open(dn)) if os.path.exists(dn) else set()


for ci, case in enumerate(cases, 1):
    if case in done:
        continue
    t0 = time.time(); root = L.case_root("hcc", case, L.HCC_ROOT); ct_path = os.path.join(root, case, "ct.nii.gz"); nii = nib.load(ct_path); ct = np.asarray(nii.dataobj).astype(np.float32)
    _, sp, seg, ann, zidx, lab, sel = case_geometry(root, case); idx = np.flatnonzero(ann); sub = lambda m: m[:, :, idx]      # noqa: E731
    crow = case_lesions(sub(ct), sub(seg == 2), sub(seg == 1), sp, exclude=sub(seg == 3)); les = {}
    for rank, (diam, cid) in enumerate(sel[:2], 1):
        if (case, rank) in table:
            m = np.zeros(seg.shape, bool); m[:, :, ann] = lab == cid; les[rank] = {"mask": m, "T": float(ct[m].mean()), "L": float(crow[rank - 1]["liver_hu"])}
    if not les:
        open(dn, "a").write(case + "\n"); continue
    ctx = eng.prepare(ct_path, EN.mapped_label(case, root, outdir, {r: v["mask"] for r, v in les.items()})); base_o = eng.predict(ctx); rows = []; toks = {}
    for rank, v in les.items():
        r, _ = L.lesion_rows("hcc", case, root, table, {"base": base_o["tumour"]}); r = next(x for x in r if x["rank"] == rank); c = v["T"] - v["L"]; at = arm_targets(v["T"], v["L"]); v["base_row"] = r
        roi = ctx.Lmap == 10 + rank; tb, nb = eng.record_tokens(ctx, roi); toks[rank] = {"base": tb, "n": nb}
        rows.append({"case": case, "rank": rank, "arm": "", "condition": "base", "T": v["T"], "L": v["L"], "c": c, "target": "", "realised_mean": "", "flagged": "", "coverage": r["coverage"], "dice_full": r["dice_full"], "dice_within_10mm": r["dice_within_10mm"], "pred_diam_mm": r["pred_diam_mm"], "n_tokens": nb})
        if at is None:
            continue
        arm, t_edit, t_ctrl = at
        if not ((arm == "A" and r["coverage"] >= 0.10) or (arm == "B" and r["coverage"] < 0.10)):
            continue
        for cond, target in (("edit", t_edit), ("control", t_ctrl)):
            edited, info = edit_lesion(ct, v["mask"], target); img = nib.Nifti1Image(edited.astype(np.float32), nii.affine, nii.header); img.set_data_dtype(np.float32); img.header.set_slope_inter(None)
            p = os.path.join(outdir, "tmp_edit.nii.gz"); nib.save(img, p); cx = eng.prepare(p, None); cx.Lmap = ctx.Lmap; assert (tuple(cx.x.shape) == tuple(ctx.x.shape)) if eng.name == "A" else (cx.data.shape == ctx.data.shape), "edited volume changed the preprocessed grid"; o = eng.predict(cx)
            rr, _ = L.lesion_rows("hcc", case, root, table, {cond: o["tumour"]}); rr = next(x for x in rr if x["rank"] == rank); tk, nk = (eng.record_tokens(cx, roi) if cond == "edit" else (None, 0))
            if tk is not None:
                toks[rank]["edit"] = tk
            rows.append({"case": case, "rank": rank, "arm": arm, "condition": cond, "T": v["T"], "L": v["L"], "c": c, "target": target, "realised_mean": info["realised_mean"], "flagged": info["flagged"], "coverage": rr["coverage"], "dice_full": rr["dice_full"],
                         "dice_within_10mm": rr["dice_within_10mm"], "pred_diam_mm": rr["pred_diam_mm"], "n_tokens": nk}); os.remove(p)
        for x in rows:
            if x["case"] == case and x["rank"] == rank and x["condition"] == "base":
                x["arm"] = arm
    for rank, t in toks.items():
        if "edit" in t:
            np.savez_compressed(os.path.join(outdir, "tokens", f"{case}_r{rank}.npz"), base=t["base"], edit=t["edit"], n=t["n"])
    new = not os.path.exists(os.path.join(outdir, "m3_lesions.csv"))
    with open(os.path.join(outdir, "m3_lesions.csv"), "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader() if new else None; w.writerows(rows)
    open(dn, "a").write(case + "\n"); print(f"[{model}] {ci}/{len(cases)} {case} {time.time() - t0:.0f}s rows {len(rows)}", flush=True)
print("M3_RUN_DONE", flush=True)
