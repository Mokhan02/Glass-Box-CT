"""W3 Stage 2, M2 (and the M4 baseline maps) runner, GPU (PREREG_stage2.md; research use only for model A). Per case of the Stage 2 cohort (primary 74 + HCC_027): one preprocessing, then the pipeline under the conditions
base (unhooked; saves the tumour decision-margin map for M4), leace, random, hu (the cross-fitted erasers of the case's fold; HCC_027 uses the `all` set) and, on the first 3 cases, identity (d = 0; must equal base in the same session). The eraser acts on the intervention tensor in every window
(A: swinViT.layers2[0] output; D: encoder.stages[3] output, mirroring ON). Outputs in <outdir>: lesions.csv (arm = <model>_<condition>, Stage 1 columns incl. dice_within_10mm), cases.csv (per case and condition: organ Dice vs the GT liver, centroid offset), maps/<case>.npz (base only: margin crop float16 in [x, y, z] order,
GT label map uint8 (0 bg, 1 liver, 2 tumour, 3 unknown), bbox offset, grid spacing [x, y, z]), determinism.json (base vs Stage 1 lesion coverage on every case; identity vs base). Resumable per case.
Usage (cwd = repo checkout, GPU): python scripts/w3s2_m2_run.py A <outdir> <erasers.npz> <repo_A> <swin weights> [--limit N] [--cases c ...] [--shard i/n]
                                   python scripts/w3s2_m2_run.py D <outdir> <erasers.npz> <nnU-Net model folder> [--limit N] [--cases c ...] [--shard i/n]"""
import csv
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg import w3s2_engines as EN  # noqa: E402
from glassbox.seg.w2_lib import case_geometry  # noqa: E402

model, outdir, erasers_path = sys.argv[1:4]; rest = sys.argv[4:]; limit = int(rest[rest.index("--limit") + 1]) if "--limit" in rest else None; only = rest[rest.index("--cases") + 1:] if "--cases" in rest else None
shard = rest[rest.index("--shard") + 1] if "--shard" in rest else None; pos = [a for a in rest if not a.startswith("--") and a not in (only or []) and a != (str(limit) if limit else None) and a != shard]
primary = sorted(l.strip() for l in open("results/w3_models/stage2/m1/cases_primary74.txt") if l.strip()); fold = {c: i % 5 for i, c in enumerate(primary)}
table = L.lesion_table("hcc", include_excluded=True); cases = sorted({k[0] for k in table}); cases = only or cases[:limit]
if shard:
    si, sn = (int(v) for v in shard.split("/")); cases = cases[si::sn]
os.makedirs(os.path.join(outdir, "maps"), exist_ok=True); dev = torch.device("cuda"); E = np.load(erasers_path)
eng = EN.EngineA(pos[0], pos[1], outdir, dev) if model == "A" else EN.EngineD(pos[0], outdir, dev)


def eraser(case, kind):
    f = fold.get(case, "all"); return {k: E[f"{f}_{kind}_{k}"] for k in ("mu", "d", "a", "c")}


def append(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader() if new else None; w.writerows(rows)


done = set(l.strip() for l in open(os.path.join(outdir, "done_cases.txt"))) if os.path.exists(os.path.join(outdir, "done_cases.txt")) else set()
stage1 = {(r["arm"], r["case"], r["rank"]): r for r in csv.DictReader(open("results/w3_models/stage1/A_swin_hcc/w3_lesions_A_swin_hcc.csv" if model == "A" else "results/w3_models/stage1/D_nnunet_hcc/w3_lesions_D_nnunet_hcc.csv"))}
s1arm = "A_swin_shipped" if model == "A" else "D_nnunet"; det = {"base_vs_stage1": [], "identity_vs_base": []}
for ci, case in enumerate(cases, 1):
    if case in done:
        continue
    t0 = time.time(); root = L.case_root("hcc", case, L.HCC_ROOT); ct = os.path.join(root, case, "ct.nii.gz"); lab = EN.mapped_label(case, root, outdir); ctx = eng.prepare(ct, lab)
    _, geo = L.lesion_rows("hcc", case, root, {}, {}); seg, ann, sp = geo["seg"], geo["ann"], geo["spacing"]; in_shape = nib.load(ct).shape
    conds = ["base", "leace", "random", "hu"] + (["identity"] if case in sorted({k[0] for k in table})[:3] else []); rows, crow, outs = [], [], {}
    for cond in conds:
        er = None if cond == "base" else (EN.identity_eraser(eng.dim) if cond == "identity" else eraser(case, cond)); o = eng.predict(ctx, er); outs[cond] = o
        r, _ = L.lesion_rows("hcc", case, root, table, {f"{model}_{cond}": o["tumour"]}); rows += r; al = L.alignment_report(o["liver"], seg, ann, sp, in_shape)
        crow.append({"case": case, "cond": cond, "fold": fold.get(case, "all"), "organ_dice": al["organ_dice"], "centroid_offset_mm": al["centroid_offset_mm"], "tumour_voxels": int(o["tumour"][:seg.shape[0], :seg.shape[1], :seg.shape[2]][:, :, ann].sum())})
    for r in rows:
        if r["arm"] == f"{model}_base":
            s = stage1.get((s1arm, case, str(r["rank"])))
            if s is not None:
                det["base_vs_stage1"].append({"case": case, "rank": r["rank"], "abs_cov_diff": abs(float(s["coverage"]) - r["coverage"]), "flag_same": (float(s["coverage"]) >= 0.1) == (r["coverage"] >= 0.1)})
        if r["arm"] == f"{model}_identity":
            b = next(x for x in rows if x["arm"] == f"{model}_base" and x["rank"] == r["rank"]); det["identity_vs_base"].append({"case": case, "rank": r["rank"], "abs_cov_diff": abs(b["coverage"] - r["coverage"])})
    # M4 baseline map: tumour decision margin on the model grid, annotated slab only, always stored in [x, y, z] order
    mg, Lm, spg = outs["base"]["margin_grid"], ctx.Lmap, tuple(ctx.grid_spacing)
    if eng.name == "D":
        mg, Lm, spg = mg.transpose(2, 1, 0), Lm.transpose(2, 1, 0), spg[::-1]
    kn = np.argwhere(Lm != 3); lo, hi = kn.min(0), kn.max(0) + 1; sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    np.savez_compressed(os.path.join(outdir, "maps", f"{case}.npz"), margin=mg[sl].astype(np.float16), label=Lm[sl].astype(np.uint8), offset=lo, spacing=np.array(spg))
    append(os.path.join(outdir, "lesions.csv"), L.FIELDS, rows); append(os.path.join(outdir, "cases.csv"), list(crow[0]), crow); open(os.path.join(outdir, "done_cases.txt"), "a").write(case + "\n")
    print(f"[{model}] {ci}/{len(cases)} {case} {time.time() - t0:.0f}s organ Dice " + " ".join(f"{c['cond']}={c['organ_dice']:.3f}" for c in crow), flush=True)
old = json.load(open(os.path.join(outdir, "determinism.json"))) if os.path.exists(os.path.join(outdir, "determinism.json")) else {"base_vs_stage1": [], "identity_vs_base": []}
for k in det:
    old[k] += det[k]
old["summary"] = {"max_base_vs_stage1_cov_diff": max((d["abs_cov_diff"] for d in old["base_vs_stage1"]), default=None), "detection_flag_mismatches": sum(not d["flag_same"] for d in old["base_vs_stage1"]), "max_identity_vs_base_cov_diff": max((d["abs_cov_diff"] for d in old["identity_vs_base"]), default=None)}
json.dump(old, open(os.path.join(outdir, "determinism.json"), "w"), indent=1); print("M2_RUN_DONE", json.dumps(old["summary"]), flush=True)
