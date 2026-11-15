"""[EXPLORATORY, W1 secondary analysis, requested by the researcher 2026-11-15; CPU only] MSD Task08 HepaticVessel (label 1 hepatic vessel, label 2 tumour; NO liver label): lesion-level contrast under the frozen W1 definition (results/w1_training_audit/PREREG.md) with the PSEUDO-LIVER fixed there: TotalSegmentator `total` task, class `liver`, full resolution (not --fast), CPU (a `--fast` flag exists ONLY to time the 3 mm model; it is not the PREREG setting); keep the largest connected component; remove all tumour voxels; lesions whose component lies entirely further than 10 mm from the pseudo-liver are excluded and counted. Every Task08 number is labelled pseudo-liver.
Stages (run in order; cwd = repo root):
  liver     <Task08 dir> <out dir> [--threads N] [--limit K] [--cases a b ...]   TotalSegmentator liver mask per labelled case -> <out>/pseudo_liver/<case>.nii.gz (skips existing)
  contrast  <Task08 dir> <out dir>                                                lesion table -> <out>/task08_lesions.csv (same columns as msd_lesions.csv plus the exclusion flag, contrast without the vessel label voxels as a sensitivity) and <out>/task08_inventory.csv
Needs the venv with TotalSegmentator (stage liver) and nibabel / scipy (stage contrast)."""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")


def names(root):
    ds = json.load(open(os.path.join(root, "dataset.json"))); return sorted(os.path.basename(t["label"]) for t in ds["training"])


def stage_liver(root, out, argv):
    import nibabel as nib
    import torch
    from totalsegmentator.python_api import totalsegmentator
    opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d      # noqa: E731
    th = int(opt("--threads", 5)); torch.set_num_threads(th); lim = int(opt("--limit", 0)) or None; cases = argv[argv.index("--cases") + 1:] if "--cases" in argv else None
    os.makedirs(os.path.join(out, "pseudo_liver"), exist_ok=True); todo = [n for n in names(root) if cases is None or n.replace(".nii.gz", "") in cases][:lim]
    for i, n in enumerate(todo, 1):
        dst = os.path.join(out, "pseudo_liver", n)
        if os.path.exists(dst):
            continue
        t0 = time.time(); img = nib.load(os.path.join(root, "imagesTr", n))
        seg = totalsegmentator(img, None, ml=True, task="total", roi_subset=["liver"], device="cpu", fast=("--fast" in argv), quiet=True); m = np.asarray(seg.dataobj) > 0
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine), dst + ".part.nii.gz"); os.replace(dst + ".part.nii.gz", dst)
        print(f"[liver] {i}/{len(todo)} {n} liver voxels {int(m.sum())} {time.time() - t0:.0f}s", flush=True)


def stage_contrast(root, out):
    import nibabel as nib
    from scipy import ndimage
    from glassbox.audit.contrast import case_lesions, select_lesions
    rows, inv = [], []
    for k, n in enumerate(names(root), 1):
        cid = n.replace(".nii.gz", ""); pl = os.path.join(out, "pseudo_liver", n)
        if not os.path.exists(pl):
            inv.append({"case": cid, "status": "no pseudo-liver mask"}); continue
        img = nib.as_closest_canonical(nib.load(os.path.join(root, "imagesTr", n))); lab = nib.as_closest_canonical(nib.load(os.path.join(root, "labelsTr", n))); liv = nib.as_closest_canonical(nib.load(pl))
        if not (img.shape == lab.shape == liv.shape):
            inv.append({"case": cid, "status": "shape mismatch"}); continue
        sp = tuple(float(z) for z in img.header.get_zooms()[:3]); ct = np.asarray(img.dataobj).astype(np.float32); L = np.asarray(lab.dataobj).astype(np.uint8); lv = np.asarray(liv.dataobj) > 0
        cc, nc = ndimage.label(lv)
        if nc == 0:
            inv.append({"case": cid, "status": "empty pseudo-liver"}); continue
        lv = cc == (1 + int(np.argmax(np.bincount(cc.ravel())[1:])))      # largest connected component
        tum = L == 2; dist = ndimage.distance_transform_edt(~lv, sampling=sp)
        lab3, sel = select_lesions(tum, sp, None)
        base = case_lesions(ct, tum, lv, sp, n_lesions=None); nov = case_lesions(ct, tum, lv, sp, n_lesions=None, exclude=(L == 1))
        far = []
        for rank, (d, i) in enumerate(sel, 1):
            far.append(float(dist[lab3 == i].min()))
        for r, b, f in zip(base, nov, far):
            rows.append({"patient": cid, "dataset": "msd_task08_pseudo_liver", **r, "contrast_hu_no_vessel_label": b["contrast_hu"], "min_dist_to_pseudo_liver_mm": f, "excluded_far_from_liver": f > 10.0})
        inv.append({"case": cid, "status": "ok", "liver_vox": int(lv.sum()), "tumour_vox": int(tum.sum()), "n_lesions_ge10mm": len(base)})
        if k % 25 == 0:
            print(f"[contrast] {k}/303", flush=True)
    for name, data in (("task08_lesions.csv", rows), ("task08_inventory.csv", inv)):
        keys = sorted({k for r in data for k in r}, key=lambda x: (list(data[0]).index(x) if x in data[0] else 99, x)); w = csv.DictWriter(open(os.path.join(out, name), "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(data)
    print(f"lesions {len(rows)} cases with lesions {len({r['patient'] for r in rows})} excluded far {sum(r['excluded_far_from_liver'] for r in rows)}")


if __name__ == "__main__":
    st, root, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    stage_liver(root, out, sys.argv[4:]) if st == "liver" else stage_contrast(root, out)
