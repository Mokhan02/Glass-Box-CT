"""W2-M route check runner (results/w2_mech/PREREG_route.md). HCC main-set lesions (UID route, minus the 10 v2 pilot lesions; 76 lesions, 68 cases) under the cross-fitted HCC mech-B erasers (leace, random, hu; fold = index of the case in the sorted
clean-89 `cache/rr` case list mod 5), at the mid-encoder hook of mech B:
  point arms (the W2 v3 arms): hcc_133_1 (label 133, point label 1; PRIMARY) and hcc_26_3 (class 26, point label 3; secondary), each x {leace, random, hu};
  positive control: class 26, AUTOMATIC route (no point), per case: unhooked, leace, random, hu, evaluated on every lesion of the case;
  liver organ check on the seeded 30-case subset: class 1 with one point at the interior of the liver containing the first main-set lesion, unhooked and under each eraser.
Stages: `check` (10 fixed lesions: unhooked point arms reproduce the W2 v3 CSV within 2e-5 and an identity hook is bit-identical; exit 1 otherwise), `time` (the first N cases with every condition, prints a projection), `run`. All predicted masks saved.
Usage: python scripts/w2m_route_run.py check|time|run <inferer_dir> <weights> <mech_b_erasers.npz> <rr_dir> <outdir> [N]"""
import csv
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.seg.w2_lib import Runner, case_geometry, dice_within, interior_point, lesion_metrics  # noqa: E402

stage, inferer_dir, weights, erasers, rr_dir, outdir = sys.argv[1:7]; N = int(sys.argv[7]) if len(sys.argv) > 7 else 0
HCC = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii/hcc_tace_seg"
E = np.load(erasers); clean = sorted(f[:-4] for f in os.listdir(rr_dir) if f.endswith(".npz") and not f.endswith(".part.npz")); fold_of = {c: i % 5 for i, c in enumerate(clean)}; assert len(clean) == 89
pilot = {(r["case"], int(r["rank"])) for r in csv.DictReader(open("results/w2_kidney/v2_pilot_selection.csv")) if r["organ"] == "hcc"}
table = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/exp8_lesions/lesion_table.csv")) if r["route"] == "uid"}
main = sorted(k for k in table if k not in pilot); cases = sorted({c for c, _ in main}); assert len(main) == 76 and len(cases) == 68 and all(c in fold_of for c in cases)
v3 = {(r["arm"], r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/v3/w2v2_main_hcc.csv"))}
subset = {r["case"] for r in csv.DictReader(open("results/w2_mech/route_organ_subset.csv"))}
ARMS = {"hcc_133_1": (133, 1), "hcc_26_3": (26, 3)}; CONDS = ["leace", "random", "hu"]
R = Runner(inferer_dir, weights)


def params(cond, case):
    f = fold_of[case]; return tuple(E[f"f{f}_{cond}_{k}"] for k in ("mu", "d", "a", "c"))      # mech B naming; the random control is mech B's seed-(100 + fold) projection through the training-token mean


def lesions_of(case):
    nii, sp, seg, ann, zidx, lab, sel = case_geometry(HCC, case); out = []
    for rank, (diam, cid) in enumerate(sel[:2], 1):
        if (case, rank) not in table or (case, rank) in pilot:
            continue
        ref = table[(case, rank)]; assert abs(float(ref["diam_mm"]) - diam) < 0.01
        comp_c = lab == cid; pt_c = interior_point(comp_c, sp); comp = np.zeros(seg.shape, bool); comp[:, :, ann] = comp_c; out.append({"rank": rank, "diam": diam, "comp_c": comp_c, "pt_c": pt_c, "pt": (pt_c[0], pt_c[1], int(zidx[pt_c[2]])), "ref": ref})
    return sp, seg, ann, zidx, out


def row(extra, m, comp_c, pred_a, sp):
    return {**extra, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}, "dice_within_10mm": round(dice_within(pred_a, comp_c, sp), 5)}


def point_pred(data, img, M, les, seg, label, pl):
    return R.predict(data, img, M, label, pl, les["pt"])[: seg.shape[0], : seg.shape[1], : seg.shape[2]]


if stage == "check":
    ok = True; picks = [main[i] for i in range(0, len(main), 8)][:10]; byc = {}
    for c, r in picks:
        byc.setdefault(c, []).append(r)
    for case, ranks in byc.items():
        sp, seg, ann, zidx, ls = lesions_of(case); data, img, M = R.prep(os.path.join(HCC, case, "ct.nii.gz"))
        for les in [l for l in ls if l["rank"] in ranks]:
            for arm, (lab, pl) in ARMS.items():
                ref = v3[(arm, case, les["rank"])]; R.set_eraser(None); p0 = point_pred(data, img, M, les, seg, lab, pl); m0 = lesion_metrics(p0[:, :, ann], les["comp_c"], sp)
                R.set_eraser((E["f0_leace_mu"], np.zeros_like(E["f0_leace_d"]), E["f0_leace_a"], float(E["f0_leace_c"]))); p1 = point_pred(data, img, M, les, seg, lab, pl); R.set_eraser(None)
                same_pt = les["pt"] == (int(ref["point_x"]), int(ref["point_y"]), int(ref["point_z"])); dc, dd = abs(m0["coverage"] - float(ref["coverage"])), abs(m0["dice_full"] - float(ref["dice_full"])); idn = bool(np.array_equal(p0, p1))
                good = dc <= 2e-5 and dd <= 2e-5 and same_pt and idn; ok &= good
                print(f"{arm} {case} r{les['rank']}: |d coverage| {dc:.1e}, |d dice| {dd:.1e}, same point {same_pt}, identity bit-identical {idn} -> {'MATCH' if good else 'MISMATCH'}", flush=True)
    print("DETERMINISM AND IDENTITY CHECK", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)

os.makedirs(os.path.join(outdir, "masks"), exist_ok=True); L_rows, A_rows, O_rows = [], [], []; t0 = time.time(); todo = cases[:N] if (stage == "time" and N) else cases
for ci, case in enumerate(todo):
    tc = time.time(); sp, seg, ann, zidx, ls = lesions_of(case); data, img, M = R.prep(os.path.join(HCC, case, "ct.nii.gz"))
    for les in ls:                                                                                       # point arms
        for arm, (lab, pl) in ARMS.items():
            for cond in CONDS:
                R.set_eraser(params(cond, case)); pred = point_pred(data, img, M, les, seg, lab, pl); m = lesion_metrics(pred[:, :, ann], les["comp_c"], sp)
                L_rows.append(row({"arm": arm, "condition": cond, "case": case, "case_idx": ci, "fold": fold_of[case], "rank": les["rank"], "group": "bright" if float(les["ref"]["contrast_hu"]) > 0 else "dark", "diam_mm": round(les["diam"], 3),
                                   "contrast_hu": float(les["ref"]["contrast_hu"])}, m, les["comp_c"], pred[:, :, ann], sp))
                np.savez_compressed(os.path.join(outdir, "masks", f"{arm}__{cond}__{case}__r{les['rank']}.npz"), pred=pred, point=np.array(les["pt"]))
    for cond in ["unhooked"] + CONDS:                                                                   # positive control: automatic class-26 route, one pass per case and condition
        R.set_eraser(None if cond == "unhooked" else params(cond, case)); pred = R.predict_auto(data, img, 26)[: seg.shape[0], : seg.shape[1], : seg.shape[2]]
        for les in ls:
            m = lesion_metrics(pred[:, :, ann], les["comp_c"], sp)
            A_rows.append(row({"arm": "hcc_26_auto", "condition": cond, "case": case, "case_idx": ci, "fold": fold_of[case], "rank": les["rank"], "group": "bright" if float(les["ref"]["contrast_hu"]) > 0 else "dark", "diam_mm": round(les["diam"], 3),
                               "contrast_hu": float(les["ref"]["contrast_hu"])}, m, les["comp_c"], pred[:, :, ann], sp))
        np.savez_compressed(os.path.join(outdir, "masks", f"auto__{cond}__{case}.npz"), pred=pred)
    if case in subset and ls:                                                                             # liver organ check
        les = ls[0]; liv_a = (seg > 0)[:, :, ann]; labk, _ = ndimage.label(liv_a, structure=np.ones((3, 3, 3))); cid = int(labk[les["pt_c"][0], les["pt_c"][1], les["pt_c"][2]]); assert cid > 0
        comp = labk == cid; pc = interior_point(comp, sp); opt = (pc[0], pc[1], int(zidx[pc[2]]))
        def organ(state):
            R.set_eraser(state); p = R.predict(data, img, M, 1, 1, opt)[: seg.shape[0], : seg.shape[1], : seg.shape[2]][:, :, ann]; return float(2 * (p & comp).sum() / max(int(p.sum()) + int(comp.sum()), 1))
        orow = {"case": case, "case_idx": ci, "fold": fold_of[case], "dice_unhooked": organ(None)}
        for cond in CONDS:
            orow[f"dice_{cond}"] = organ(params(cond, case))
        O_rows.append({k: (round(v, 5) if isinstance(v, float) else v) for k, v in orow.items()})
    R.set_eraser(None)
    if stage == "time":
        print(f"timing: case {case} took {time.time() - tc:.0f}s", flush=True)
    elif ci % 10 == 0:
        print(f"[route] {ci + 1}/{len(todo)} {case} elapsed {time.time() - t0:.0f}s", flush=True)
for name, rr in (("w2m_route_lesions.csv", L_rows), ("w2m_route_auto.csv", A_rows), ("w2m_route_organ.csv", O_rows)):
    if rr:
        with open(os.path.join(outdir, name if stage == "run" else "time_" + name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
if stage == "time":
    per = (time.time() - t0) / max(len(todo), 1); print(f"TIMING PROJECTION: {per:.0f}s per case over {len(todo)} cases -> {per * 68 / 3600:.2f} h for all 68 cases", flush=True)
else:
    print(f"DONE route: {len(L_rows)} point-arm predictions, {len(A_rows)} automatic lesion-condition rows, {len(O_rows)} organ-check cases, {time.time() - t0:.0f}s", flush=True)
