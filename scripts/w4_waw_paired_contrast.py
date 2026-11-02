"""W4 scouting (CPU): per-tumour contrast on BOTH phases for every WAW-TACE patient with an arterial and a portal-venous series and a tumour mask on one of them (results/w4_waw/SCOUTING.md, results/w4_waw/CONTRAST_RULES.md).
Per patient: the mask drawn on phase m (1 arterial or 2 portal venous) is propagated to the other phase t by the rigid-then-affine registration of scripts/w4_waw_register.py (BSpline only if the affine liver Dice is below 0.92), with the component labels carried
(nearest neighbour) so that every tumour keeps its identity. Per tumour (26-connected component of at least 10 mm longest axial diameter on the drawn phase): contrast on the drawn phase and on the other phase with the project's shared function
(glassbox.audit.contrast, 5 to 20 mm shell, reference liver = TotalSegmentator label 5 of that phase minus ALL tumour voxels of that phase). Sensitivity columns: the liver eroded by 3 mm (mean contrast) and the median instead of the mean of the shell.
Registration QA per patient (pass = ALL of: liver Dice >= 0.92; >= 90% of the propagated tumour volume inside the target liver; propagated/original tumour volume in 0.85 to 1.15) is written next to the tumours; failures are excluded downstream and counted.
Usage (python -I): python -I scripts/w4_waw_paired_contrast.py <ct root> <tumour mask dir> <organ mask dir> <metadata csv> <out dir> [--workers 3] [--patients 14 21 ...]"""
import csv
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
_spec = importlib.util.spec_from_file_location("w4reg", os.path.join(HERE, "w4_waw_register.py")); REG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(REG)
from glassbox.audit.contrast import SHELL_MM, lesion_contrast, select_lesions  # noqa: E402
from glassbox.audit.waw import reference_parenchyma  # noqa: E402

QA_DICE, QA_INSIDE, QA_VOL = 0.92, 0.90, (0.85, 1.15)
ARGS = {}


def arr(img):
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0)      # [x, y, z]


FOV_MIN = 0.95      # a tumour is "outside the scanned field of view" if fewer than 95% of its voxel centres map inside the target scan (results/w4_waw/CONTRAST_RULES.md)


def fov_inside_fraction(points_idx, mov_ct, fix_ct, inv, max_pts=3000):
    """Fraction of the tumour's voxel centres (a deterministic subsample of up to max_pts; points_idx = [x, y, z] voxel indices on the moving grid) that the registration maps inside the physical extent of the target scan.
    `inv` maps moving physical points to fixed physical points (the inverse of the affine stage)."""
    if len(points_idx) == 0:
        return 0.0
    step = max(len(points_idx) // max_pts, 1); pts = points_idx[::step]; inside = 0; size = np.array(fix_ct.GetSize(), float)
    for ix in pts:
        q = inv.TransformPoint(mov_ct.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in ix))); c = np.array(fix_ct.TransformPhysicalPointToContinuousIndex(q))
        inside += bool(np.all(c >= -0.5) and np.all(c <= size - 0.5))
    return inside / len(pts)


def median_contrast(ct, lesion, liver, spacing):
    """Same shell as the shared function but the MEDIAN of the shell (sensitivity); falls back like the shared function when the shell is small."""
    sx, sy, sz = spacing; liver = liver & ~lesion
    xs, ys, zs = np.nonzero(lesion); pad = int(25 / min(sx, sy, sz)); X, Y, Z = ct.shape
    x0, x1, y0, y1, z0, z1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, X), max(ys.min() - pad, 0), min(ys.max() + pad + 1, Y), max(zs.min() - pad, 0), min(zs.max() + pad + 1, Z)
    dist = ndimage.distance_transform_edt(~lesion[x0:x1, y0:y1, z0:z1], sampling=(sx, sy, sz)); shell = liver[x0:x1, y0:y1, z0:z1] & (dist >= SHELL_MM[0]) & (dist <= SHELL_MM[1])
    L = float(np.median(ct[liver])) if shell.sum() < 200 else float(np.median(ct[x0:x1, y0:y1, z0:z1][shell]))
    return float(np.median(ct[lesion])) - L


def _process(job):
    pid, ph = job; t0 = time.time(); ct_root, tdir, odir = ARGS["ct"], ARGS["tdir"], ARGS["odir"]; pt = 2 if ph == 1 else 1
    mov_ct, fix_ct = REG.read(REG.find_ct(ct_root, pid, ph), sitk.sitkFloat32), REG.read(REG.find_ct(ct_root, pid, pt), sitk.sitkFloat32)
    lv = lambda p: REG.to_grid(sitk.BinaryThreshold(REG.read(glob.glob(os.path.join(odir, pid, f"{pid}_{p}_total_segmentator*"))[0]), REG.LIVER, REG.LIVER, 1, 0), {ph: mov_ct, pt: fix_ct}[p])      # noqa: E731
    lab_organ = lambda p: arr(REG.to_grid(REG.read(glob.glob(os.path.join(odir, pid, f"{pid}_{p}_total_segmentator*"))[0]), {ph: mov_ct, pt: fix_ct}[p]))      # noqa: E731
    org_m, org_f = lab_organ(ph), lab_organ(pt); ml, fl = lv(ph), lv(pt); tum = None
    for f in sorted(glob.glob(os.path.join(tdir, "*", pid, f"{pid}_{ph}_*_tumor_seg.nrrd"))):
        t = REG.to_grid(sitk.BinaryThreshold(REG.read(f), 1, 32767, 1, 0), mov_ct); tum = t if tum is None else sitk.Or(tum, t)
    final, qa = REG.register(fix_ct, mov_ct, fl, ml, QA_DICE)
    sp_m, sp_f = mov_ct.GetSpacing(), fix_ct.GetSpacing(); tb = arr(tum) > 0; lab, sel = select_lesions(tb, sp_m, n_lesions=None, min_diam_mm=10.0)
    lab_img = sitk.GetImageFromArray(lab.transpose(2, 1, 0).astype(np.uint16)); lab_img.CopyInformation(mov_ct)
    prop = arr(sitk.Resample(lab_img, fix_ct, final, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)); fl_a = arr(fl) > 0; ml_a = arr(ml) > 0; ct_m, ct_f = arr(mov_ct), arr(fix_ct)
    aff = sitk.CompositeTransform(final).GetNthTransform(0)      # the affine stage (index 0 of the rigid-affine result and of the affine-plus-BSpline composite)
    inv = aff.GetInverse(); n_comp = int(lab.max()); fov = {i: fov_inside_fraction(np.argwhere(lab == i), mov_ct, fix_ct, inv) for i in range(1, n_comp + 1)}; clipped = [i for i, f in fov.items() if f < FOV_MIN]
    tb_in = (lab > 0) & ~np.isin(lab, clipped); pr_in = (prop > 0) & ~np.isin(prop, clipped)      # QA criteria 2 and 3 are evaluated on the tumours inside the scanned field of view only
    pv, ov = float(pr_in.sum() * np.prod(sp_f)), float(tb_in.sum() * np.prod(sp_m)); inside = float((pr_in & fl_a).sum() / max(pr_in.sum(), 1)) if ov else float("nan"); ratio = pv / ov if ov else float("nan")
    d = qa["liver_dice"][qa["final_transform"]]; ok = bool(ov > 0 and d >= QA_DICE and inside >= QA_INSIDE and QA_VOL[0] <= ratio <= QA_VOL[1])
    ero = lambda m, sp: ndimage.distance_transform_edt(m, sampling=sp) > 3.0      # noqa: E731
    ml_e, fl_e = ml_a & ero(ml_a, sp_m), fl_a & ero(fl_a, sp_f); ref_m, ref_f = reference_parenchyma(org_m, lab > 0), reference_parenchyma(org_f, prop > 0); ref_me, ref_fe = ml_e & ref_m, fl_e & ref_f; rows = []      # label 5 minus all tumour voxels minus label 64; the eroded variants intersect the same reference with the 3 mm eroded liver
    for rank, (diam, i) in enumerate(sel, 1):
        cm_, cf_ = lab == i, prop == i; n_f = int(cf_.sum()); r = {"patient": pid, "mask_phase": ph, "target_phase": pt, "rank": rank, "diam_mm": round(diam, 2), "volume_mm3_drawn": round(float(cm_.sum() * np.prod(sp_m)), 1), "volume_mm3_propagated": round(n_f * float(np.prod(sp_f)), 1)}
        cd = lesion_contrast(ct_m, cm_, ref_m, sp_m); r["c_drawn"] = round(cd["contrast_hu"], 2); r["c_drawn_eroded"] = round(lesion_contrast(ct_m, cm_, ref_me, sp_m)["contrast_hu"], 2); r["c_drawn_median"] = round(median_contrast(ct_m, cm_, ref_m, sp_m), 2)
        if n_f >= 50:
            r["c_other"] = round(lesion_contrast(ct_f, cf_, ref_f, sp_f)["contrast_hu"], 2); r["c_other_eroded"] = round(lesion_contrast(ct_f, cf_, ref_fe, sp_f)["contrast_hu"], 2); r["c_other_median"] = round(median_contrast(ct_f, cf_, ref_f, sp_f), 2)
        else:
            r["c_other"] = r["c_other_eroded"] = r["c_other_median"] = ""
        r["c_arterial"], r["c_portal"] = (r["c_drawn"], r["c_other"]) if ph == 1 else (r["c_other"], r["c_drawn"]); r["c_arterial_eroded"], r["c_portal_eroded"] = (r["c_drawn_eroded"], r["c_other_eroded"]) if ph == 1 else (r["c_other_eroded"], r["c_drawn_eroded"])
        r["c_arterial_median"], r["c_portal_median"] = (r["c_drawn_median"], r["c_other_median"]) if ph == 1 else (r["c_other_median"], r["c_drawn_median"]); r["qa_pass"] = int(ok)
        r["fov_inside_fraction"] = round(fov[i], 4); r["category"] = "F" if fov[i] < FOV_MIN else ("K" if ok else "Q"); rows.append(r)      # F = propagated tumour outside the scanned field of view, Q = registration QA failed, K = kept
    cd0, cd1 = REG.centroid_mm(tum), REG.centroid_mm(sitk.Resample(tum, fix_ct, final, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
    q = {"patient": pid, "mask_phase": ph, "target_phase": pt, "liver_dice_none": qa["liver_dice"]["none"], "liver_dice_rigid": qa["liver_dice"]["rigid"], "liver_dice_final": d, "transform": qa["final_transform"], "propagated_inside_target_liver": inside, "volume_ratio": ratio,
         "tumour_centroid_shift_mm": None if cd1 is None else float(np.linalg.norm(cd1 - cd0)), "n_lesions_ge10mm": len(sel), "n_lesions_fov_clipped": sum(1 for _, i in sel if fov[i] < FOV_MIN), "n_components_fov_clipped": len(clipped), "qa_pass": int(ok), "seconds": round(time.time() - t0, 1), "drawn_inside_own_liver": float((tb & ml_a).sum() / max(tb.sum(), 1)), "error": ""}      # drawn_inside_own_liver separates a QA failure caused by the mask lying outside the TotalSegmentator liver from a registration failure (reported; not a pass criterion)
    return q, rows


def process(job):
    """Never let one patient stop the cohort run: any failure (for example an empty TotalSegmentator liver) becomes a QA-fail row with the reason, and the patient's tumours are excluded and counted."""
    pid, ph = job; t0 = time.time()
    try:
        return _process(job)
    except Exception as e:      # noqa: BLE001
        return ({"patient": pid, "mask_phase": ph, "target_phase": 2 if ph == 1 else 1, "liver_dice_none": "", "liver_dice_rigid": "", "liver_dice_final": "", "transform": "", "propagated_inside_target_liver": "", "volume_ratio": "", "tumour_centroid_shift_mm": "",
                 "n_lesions_ge10mm": 0, "n_lesions_fov_clipped": 0, "n_components_fov_clipped": 0, "qa_pass": 0, "seconds": round(time.time() - t0, 1), "drawn_inside_own_liver": "", "error": f"{type(e).__name__}: {str(e)[:120]}"}, [])


def init(a):
    ARGS.update(a)


def main():
    ct, tdir, odir, meta, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True); workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else max((os.cpu_count() or 4) - 2, 1)
    only = sys.argv[sys.argv.index("--patients") + 1:] if "--patients" in sys.argv else None; P = {}
    for r in csv.DictReader(open(meta)):
        P.setdefault(r["patient_id"], {})[int(r["ct_phase"])] = int(r["tumor_count"])
    jobs = sorted(((p, 1 if v.get(1, 0) > 0 else 2) for p, v in P.items() if 1 in v and 2 in v and (v.get(1, 0) > 0 or v.get(2, 0) > 0) and (not only or p in only)), key=lambda j: int(j[0]))
    print(len(jobs), "patients", flush=True); qas, rows = [], []
    with mp.get_context("spawn").Pool(workers, initializer=init, initargs=({"ct": ct, "tdir": tdir, "odir": odir},)) as pool:
        for k, (q, r) in enumerate(pool.imap_unordered(process, jobs), 1):
            qas.append(q); rows += r; print(k, "/", len(jobs), q["patient"], (f"dice {q['liver_dice_final']:.3f} inside {q['propagated_inside_target_liver']:.3f} ratio {q['volume_ratio']:.3f}" if q["error"] == "" else "ERROR " + q["error"]), f"pass {q['qa_pass']} lesions {q['n_lesions_ge10mm']} {q['seconds']}s", flush=True)
    for name, data in (("patient_qa.csv", sorted(qas, key=lambda q: int(q["patient"]))), ("tumours_paired.csv", sorted(rows, key=lambda r: (int(r["patient"]), r["rank"])))):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    json.dump({"patients": len(qas), "qa_pass": sum(q["qa_pass"] for q in qas), "tumours": len(rows)}, open(os.path.join(out, "summary.json"), "w"))


if __name__ == "__main__":
    main()
