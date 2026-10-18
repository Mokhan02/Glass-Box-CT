"""W2-M M2 runner (results/w2_mech/PREREG.md; M1 verdict = ORGAN-SPECIFIC, so the primary eraser is the kidney-native rank-1 LEACE fitted on the training folds and applied to the held-out patients).
Stage `check`: 10 fixed main-set lesions (sorted by (case, rank), indices 0, 21, 42, ...); the unhooked kidney arm must reproduce the W2 v3 CSV (dice_full and coverage within 2e-5, same point) and an identity hook (d = 0) must reproduce the unhooked prediction; exit 1 otherwise.
Stage `run`: every main-set kidney lesion (209) under kidney-native LEACE, random (seed 1000 + case index, through the training-token mean) and HU-concept erasers (fold = case index mod 5), plus the HCC-direction LEACE (secondary);
the kidney organ check (classes 5 and 14, one point at the interior of the kidney containing the lesion, class chosen per case on the UNHOOKED run, then fixed) on the seeded 60-case subset, unhooked and under each hook. All predicted masks saved.
The kidney arm is label 133 + point label 1 at the lesion's interior point. Usage: python scripts/w2m_m2_run.py check|run <inferer_dir> <weights> <erasers.npz> <outdir>"""
import csv
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.seg.w2_lib import Runner, case_geometry, interior_point, lesion_metrics  # noqa: E402

stage, inferer_dir, weights, erasers, outdir = sys.argv[1:6]
KITS = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii/c4kc_kits"
E = np.load(erasers)
pilot = {(r["case"], int(r["rank"])) for r in csv.DictReader(open("results/w2_kidney/v2_pilot_selection.csv")) if r["organ"] == "kidney"}
table = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/gt_lesions.csv")) if int(r["rank"]) <= 2}
main = sorted(k for k in table if k not in pilot); cases = sorted({c for c, _ in main}); assert len(cases) == 201 and len(main) == 209
v3 = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/v3/w2v2_main_kidney.csv")) if r["arm"] == "kidney_133_1"}
subset = {r["case"] for r in csv.DictReader(open("results/w2_mech/m2_organ_subset.csv"))}
CONDS = ["leace_native", "random_native", "hu_native", "leace_hcc"]
R = Runner(inferer_dir, weights)


def params(cond, case_idx):
    f = case_idx % 5
    if cond == "leace_native":
        return tuple(E[f"native_f{f}_leace_{k}"] for k in ("mu", "d", "a", "c"))
    if cond == "hu_native":
        return tuple(E[f"native_f{f}_hu_{k}"] for k in ("mu", "d", "a", "c"))
    if cond == "random_native":
        v = np.random.default_rng(1000 + case_idx).normal(size=E[f"native_f{f}_leace_mu"].shape); v /= np.linalg.norm(v); return (E[f"native_f{f}_leace_mu"], v, v, 1.0)      # orthogonal projection through the training-token mean
    if cond == "leace_hcc":
        return tuple(E[f"hcc_leace_{k}"] for k in ("mu", "d", "a", "c"))
    raise KeyError(cond)


def lesion_prep(case):
    nii, sp, seg, ann, zidx, lab, sel = case_geometry(KITS, case); out = []
    for rank, (diam, cid) in enumerate(sel[:2], 1):
        if (case, rank) not in table or (case, rank) in pilot:
            continue
        ref = table[(case, rank)]; assert abs(float(ref["diam_mm"]) - diam) < 0.01
        comp_c = lab == cid; pt_c = interior_point(comp_c, sp); comp = np.zeros(seg.shape, bool); comp[:, :, ann] = comp_c; pt = (pt_c[0], pt_c[1], int(zidx[pt_c[2]]))
        out.append({"rank": rank, "diam": diam, "comp_c": comp_c, "comp": comp, "pt": pt, "pt_c": pt_c, "ref": ref})
    return nii, sp, seg, ann, zidx, out


def run_lesion(data, img, M, les, seg, ann, sp, cond, state):
    R.set_eraser(state); pred = R.predict(data, img, M, 133, 1, les["pt"])[: seg.shape[0], : seg.shape[1], : seg.shape[2]]
    return pred, lesion_metrics(pred[:, :, ann], les["comp_c"], sp)


if stage == "check":
    ok = True; picks = [main[i] for i in range(0, len(main), 21)][:10]; byc = {}
    for c, r in picks:
        byc.setdefault(c, []).append(r)
    for case, ranks in byc.items():
        nii, sp, seg, ann, zidx, ls = lesion_prep(case); data, img, M = R.prep(os.path.join(KITS, case, "ct.nii.gz"))
        for les in [l for l in ls if l["rank"] in ranks]:
            ref = v3[(case, les["rank"])]
            p0, m0 = run_lesion(data, img, M, les, seg, ann, sp, None, None)
            ident = (E["native_f0_leace_mu"], np.zeros_like(E["native_f0_leace_d"]), E["native_f0_leace_a"], 1.0)
            p1, m1 = run_lesion(data, img, M, les, seg, ann, sp, None, ident)
            same_pt = les["pt"] == (int(ref["point_x"]), int(ref["point_y"]), int(ref["point_z"]))
            d_cov, d_dice = abs(m0["coverage"] - float(ref["coverage"])), abs(m0["dice_full"] - float(ref["dice_full"])); id_same = bool(np.array_equal(p0, p1))
            good = d_cov <= 2e-5 and d_dice <= 2e-5 and same_pt and id_same; ok &= good
            print(f"{case} r{les['rank']}: |d coverage| {d_cov:.1e}, |d dice| {d_dice:.1e}, same point {same_pt}, identity hook equals unhooked {id_same} -> {'MATCH' if good else 'MISMATCH'}", flush=True)
    print("DETERMINISM AND IDENTITY CHECK", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)

os.makedirs(os.path.join(outdir, "masks"), exist_ok=True); rows, orows = [], []; t0 = time.time()
for ci, case in enumerate(cases):
    nii, sp, seg, ann, zidx, ls = lesion_prep(case); data, img, M = R.prep(os.path.join(KITS, case, "ct.nii.gz"))
    for les in ls:
        for cond in CONDS:
            pred, m = run_lesion(data, img, M, les, seg, ann, sp, cond, params(cond, ci))
            rows.append({"condition": cond, "case": case, "case_idx": ci, "fold": ci % 5, "rank": les["rank"], "group": "bright" if float(les["ref"]["contrast_hu"]) > 0 else "dark", "diam_mm": round(les["diam"], 3),
                         "contrast_hu": float(les["ref"]["contrast_hu"]), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}})
            np.savez_compressed(os.path.join(outdir, "masks", f"{cond}__{case}__r{les['rank']}.npz"), pred=pred, point=np.array(les["pt"]))
    if case in subset and ls:                                              # kidney organ check: the kidney containing the first main-set lesion
        les = ls[0]; kid_a = (seg > 0)[:, :, ann]; labk, _ = ndimage.label(kid_a, structure=np.ones((3, 3, 3))); cid = int(labk[les["pt_c"][0], les["pt_c"][1], les["pt_c"][2]]); assert cid > 0
        comp = labk == cid; pc = interior_point(comp, sp); opt = (pc[0], pc[1], int(zidx[pc[2]]))
        od = lambda p: float(2 * (p[:, :, ann] & comp).sum() / max(int(p[:, :, ann].sum()) + int(comp.sum()), 1))
        def organ(cls, state):
            R.set_eraser(state); return od(R.predict(data, img, M, cls, 1, opt)[: seg.shape[0], : seg.shape[1], : seg.shape[2]])
        d5, d14 = organ(5, None), organ(14, None); sel_cls = 5 if d5 >= d14 else 14                       # class chosen on the UNHOOKED run, then fixed
        orow = {"case": case, "case_idx": ci, "fold": ci % 5, "organ_class": sel_cls, "dice_unhooked": max(d5, d14), "dice_unhooked_class5": d5, "dice_unhooked_class14": d14}
        for cond in CONDS:
            orow[f"dice_{cond}"] = organ(sel_cls, params(cond, ci))
        orows.append({k: (round(v, 5) if isinstance(v, float) else v) for k, v in orow.items()})
    R.set_eraser(None)
    if ci % 10 == 0:
        print(f"[run] {ci + 1}/{len(cases)} {case} elapsed {time.time() - t0:.0f}s", flush=True)
        for name, rr in (("w2m_m2_lesions.csv", rows), ("w2m_m2_organ.csv", orows)):
            if rr:
                with open(os.path.join(outdir, name), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
for name, rr in (("w2m_m2_lesions.csv", rows), ("w2m_m2_organ.csv", orows)):
    with open(os.path.join(outdir, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
print(f"DONE run: {len(rows)} lesion-condition predictions, {len(orows)} organ-check cases, {time.time() - t0:.0f}s", flush=True)
