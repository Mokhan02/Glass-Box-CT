"""W4 scouting (CPU): can a WAW-TACE tumour mask be propagated from the phase it was drawn on to the other phase (arterial <-> portal venous)? (results/w4_waw/SCOUTING.md)
Per patient: moving = the phase the mask was drawn on, fixed = the other phase. Rigid (Euler3D) then affine, Mattes mutual information, three resolution levels, on a 3 mm grid cropped around the liver,
metric restricted to the fixed liver (TotalSegmentator label 5, dilated 20 mm); initial translation = liver centroid difference. A BSpline stage is run ONLY for a patient whose affine liver Dice is below --deform-below.
QA written per patient: liver Dice (moving liver on the fixed grid) with no registration / rigid / affine (/ deformable), tumour centroid shift (mm, original centroid to propagated centroid in physical space), fraction of the propagated tumour inside the fixed liver,
propagated / original tumour volume, and an overlay PNG (fixed phase with the propagated mask and the fixed liver outline, moving phase with the original mask).
Patients: a seeded sample (SHA-256 of the name `w4_waw_register`) of 3 patients masked on the arterial and 2 on the portal-venous phase, among those with both phases and a mask there.
Usage (python -I): python -I scripts/w4_waw_register.py <ct root> <tumour mask dir> <organ mask dir> <metadata csv> <out dir> [--patients 10 527 ...] [--deform-below 0.9]"""
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.getcwd())
LIVER = 5


def read(path, pixel=None):
    im = sitk.ReadImage(path)
    return sitk.Cast(im, pixel) if pixel else im


def find_ct(root, pid, ph):
    h = glob.glob(os.path.join(root, "**", f"{pid}_{ph}_scan.nii*"), recursive=True)
    return h[0] if h else None


def to_grid(mask_img, ref):
    """Mask onto the reference grid (nearest neighbour, physical space)."""
    return sitk.Resample(mask_img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def dice(a, b):
    a = sitk.GetArrayFromImage(a) > 0; b = sitk.GetArrayFromImage(b) > 0; s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def centroid_mm(mask_img):
    arr = sitk.GetArrayFromImage(mask_img) > 0
    if not arr.any():
        return None
    idx = np.argwhere(arr).mean(0)[::-1]          # x, y, z in index units
    return np.array(mask_img.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in idx)))


def crop_to(mask_img, margin_mm=40.0):
    bb = sitk.LabelShapeStatisticsImageFilter(); bb.Execute(mask_img); x, y, z, sx, sy, sz = bb.GetBoundingBox(1); sp = mask_img.GetSpacing()
    lo = [max(int(v - margin_mm / s), 0) for v, s in zip((x, y, z), sp)]; hi = [min(int(v + w + margin_mm / s), n) for v, w, s, n in zip((x, y, z), (sx, sy, sz), sp, mask_img.GetSize())]
    return lo, [h - l for h, l in zip(hi, lo)]


def downsample(img, spacing=3.0, interp=sitk.sitkLinear, ref_box=None):
    sp = img.GetSpacing(); size = [int(round(n * s / spacing)) for n, s in zip(img.GetSize(), sp)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), [spacing] * 3, img.GetDirection(), -1024 if interp == sitk.sitkLinear else 0, img.GetPixelID())


def register(fixed_ct, moving_ct, fixed_liver, moving_liver, deform_below=0.9):
    t0 = time.time()
    lo, size = crop_to(fixed_liver); roi = lambda im: sitk.RegionOfInterest(im, size, lo)      # noqa: E731
    f = roi(fixed_ct); fl = roi(fixed_liver)
    # moving image resampled onto the (cropped) fixed grid's neighbourhood is not needed: SimpleITK evaluates the moving image in physical space; it only has to cover the liver
    f3 = downsample(sitk.Clamp(sitk.Cast(f, sitk.sitkFloat32), sitk.sitkFloat32, -200, 300)); m3 = sitk.Clamp(sitk.Cast(moving_ct, sitk.sitkFloat32), sitk.sitkFloat32, -200, 300)
    fmask = sitk.BinaryDilate(downsample(fl, interp=sitk.sitkNearestNeighbor), [int(round(20 / 3.0))] * 3)
    cf, cm = centroid_mm(fixed_liver), centroid_mm(moving_liver)
    init = sitk.TranslationTransform(3, tuple(float(v) for v in (cm - cf)))      # maps fixed points to moving points
    out = {"initial_translation_mm": [float(v) for v in (cm - cf)]}

    def run(tx, kind):
        R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(50); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, 1); R.SetMetricFixedMask(fmask)
        R.SetInterpolator(sitk.sitkLinear); R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetOptimizerAsRegularStepGradientDescent(2.0 if kind == "rigid" else 1.0, 1e-4, 200, relaxationFactor=0.6); R.SetOptimizerScalesFromPhysicalShift(); R.SetInitialTransform(tx, inPlace=False)
        return R.Execute(f3, m3)

    rigid0 = sitk.Euler3DTransform(); rigid0.SetTranslation(tuple(float(v) for v in (cm - cf))); rigid0.SetCenter(tuple(float(v) for v in cf))
    rigid = run(rigid0, "rigid")
    try:      # Execute() returns the optimised transform, wrapped in a composite when the initial transform was not set in place
        p = sitk.Euler3DTransform(sitk.CompositeTransform(rigid).GetNthTransform(0))
    except RuntimeError:
        p = sitk.Euler3DTransform(rigid)      # the affine stage starts from the optimised rigid transform
    A = sitk.AffineTransform(3); A.SetCenter(p.GetCenter()); A.SetMatrix(p.GetMatrix()); A.SetTranslation(p.GetTranslation())
    affine = run(A, "affine")
    tx = {"none": sitk.Transform(), "rigid": rigid, "affine": affine}

    def liver_dice(t):
        return dice(sitk.Resample(moving_liver, fixed_liver, t, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8), fixed_liver)

    out["liver_dice"] = {k: liver_dice(t) for k, t in tx.items()}
    final, name = affine, "affine"
    if out["liver_dice"]["affine"] < deform_below:
        bs = sitk.BSplineTransformInitializer(f3, [6, 6, 6]); R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(50); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, 1); R.SetMetricFixedMask(fmask)
        R.SetInterpolator(sitk.sitkLinear); R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=60); R.SetShrinkFactorsPerLevel([2, 1]); R.SetSmoothingSigmasPerLevel([1, 0]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetMovingInitialTransform(affine); R.SetInitialTransform(bs, inPlace=False); d = R.Execute(f3, m3); comp = sitk.CompositeTransform(affine); comp.AddTransform(d)
        out["liver_dice"]["deformable"] = liver_dice(comp); final, name = comp, "deformable"
    out["final_transform"] = name; out["seconds"] = round(time.time() - t0, 1)
    return final, out


def pick_patients(meta_csv, tdir, k_art=3, k_pv=2):
    P = {}
    for r in csv.DictReader(open(meta_csv)):
        P.setdefault(r["patient_id"], {})[int(r["ct_phase"])] = int(r["tumor_count"])
    art = sorted(p for p, v in P.items() if 1 in v and 2 in v and v[1] > 0); pv = sorted(p for p, v in P.items() if 1 in v and 2 in v and v[2] > 0 and v.get(1, 0) == 0)
    rng = random.Random(int(hashlib.sha256(b"w4_waw_register").hexdigest()[:8], 16))
    return [(p, 1) for p in rng.sample(art, k_art)] + [(p, 2) for p in rng.sample(pv, k_pv)]


def main():
    ctroot, tdir, odir, meta, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
    only = sys.argv[sys.argv.index("--patients") + 1:] if "--patients" in sys.argv else None
    thr = float(sys.argv[sys.argv.index("--deform-below") + 1]) if "--deform-below" in sys.argv else 0.9
    todo = [(p, ph) for p, ph in pick_patients(meta, tdir) if not only or p in only]; res = []
    for pid, ph in todo:
        pt = 2 if ph == 1 else 1
        mf = sorted(glob.glob(os.path.join(tdir, "*", pid, f"{pid}_{ph}_*_tumor_seg.nrrd")))
        mov_ct, fix_ct = read(find_ct(ctroot, pid, ph), sitk.sitkFloat32), read(find_ct(ctroot, pid, pt), sitk.sitkFloat32)
        ml = to_grid(sitk.BinaryThreshold(read(glob.glob(os.path.join(odir, pid, f"{pid}_{ph}_total_segmentator*"))[0]), LIVER, LIVER, 1, 0), mov_ct)
        fl = to_grid(sitk.BinaryThreshold(read(glob.glob(os.path.join(odir, pid, f"{pid}_{pt}_total_segmentator*"))[0]), LIVER, LIVER, 1, 0), fix_ct)
        tum = None
        for f in mf:
            t = to_grid(sitk.BinaryThreshold(read(f), 1, 32767, 1, 0), mov_ct); tum = t if tum is None else sitk.Or(tum, t)
        final, qa = register(fix_ct, mov_ct, fl, ml, thr)
        prop = sitk.Resample(tum, fix_ct, final, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        c0, c1 = centroid_mm(tum), centroid_mm(prop); vox0 = float(np.prod(tum.GetSpacing())); vox1 = float(np.prod(fix_ct.GetSpacing()))
        pa, fa = sitk.GetArrayFromImage(prop) > 0, sitk.GetArrayFromImage(fl) > 0
        qa.update({"patient": pid, "mask_phase": ph, "target_phase": pt, "mask_files": len(mf), "tumour_volume_cm3_original": float((sitk.GetArrayFromImage(tum) > 0).sum() * vox0 / 1000), "tumour_volume_cm3_propagated": float(pa.sum() * vox1 / 1000),
                   "tumour_centroid_shift_mm": None if c1 is None else float(np.linalg.norm(c1 - c0)), "propagated_inside_fixed_liver": float((pa & fa).sum() / max(pa.sum(), 1)), "propagated_voxels": int(pa.sum())})
        res.append(qa); print(json.dumps(qa), flush=True)
        # overlay
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = int(round(np.argwhere(pa).mean(0)[0])) if pa.any() else pa.shape[0] // 2; zm = int(round(np.argwhere(sitk.GetArrayFromImage(tum) > 0).mean(0)[0]))
        fa_img, ma_img = sitk.GetArrayFromImage(fix_ct), sitk.GetArrayFromImage(mov_ct); fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
        ax[0].imshow(ma_img[zm], cmap="gray", vmin=-100, vmax=250); ax[0].contour(sitk.GetArrayFromImage(tum)[zm] > 0, colors="r", linewidths=0.8); ax[0].contour(sitk.GetArrayFromImage(ml)[zm] > 0, colors="lime", linewidths=0.5); ax[0].set_title(f"{pid} phase {ph} (mask drawn here)", fontsize=9)
        ax[1].imshow(fa_img[z], cmap="gray", vmin=-100, vmax=250); ax[1].contour(pa[z], colors="r", linewidths=0.8); ax[1].contour(fa[z], colors="lime", linewidths=0.5); ax[1].set_title(f"phase {pt}: propagated mask (red), TotalSegmentator liver (green); liver Dice {qa['liver_dice'][qa['final_transform']]:.3f}", fontsize=8)
        for a in ax:
            a.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"overlay_{pid}_{ph}to{pt}.png"), dpi=110); plt.close(fig)
    json.dump(res, open(os.path.join(out, "registration_qa.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
