"""W1, MSD Task03 Liver (label 1 liver, label 2 tumour; PREREG, DEVIATIONS #4). Two stages like scripts/w1_crml_extract.py:
  inventory  check every labelled case, apply the exclusion criteria, write msd_inventory.csv (NO contrast computed).
  contrast   read the inventory, compute lesion-level contrast for included cases, write msd_lesions.csv.
Reference = label 1 (liver, tumour is label 2 and therefore excluded; Task03 has no vessel labels), the same single-valued construction as HCC.
Usage: python scripts/w1_msd_extract.py inventory|contrast <Task03_Liver dir> <outdir>"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.audit.contrast import case_lesions  # noqa: E402

stage, root, outdir = sys.argv[1:4]


def load(name):
    img = nib.as_closest_canonical(nib.load(os.path.join(root, "imagesTr", name)))
    lab = nib.as_closest_canonical(nib.load(os.path.join(root, "labelsTr", name)))
    return img, lab


def inventory_one(name):
    row = {"case": name.replace(".nii.gz", ""), "included": False, "reason": "", "shape": "", "spacing": "", "label_values": "", "liver_vox": 0, "tumour_vox": 0,
           "tumour_near_liver": np.nan, "tumour_over_liver_vol": np.nan, "liver_hu_ex_tumour": np.nan, "affine_max_diff": np.nan}
    try:
        img, lab = load(name)
        row["shape"] = "x".join(map(str, img.shape)); row["spacing"] = "x".join(f"{z:.3f}" for z in img.header.get_zooms()[:3])
        row["affine_max_diff"] = float(np.abs(img.affine - lab.affine).max())
        if img.shape != lab.shape or row["affine_max_diff"] > 1e-3:
            row["reason"] = "image/label geometry mismatch"; return row
        L = np.asarray(lab.dataobj).astype(np.uint8); vals = sorted(int(v) for v in np.unique(L)); row["label_values"] = ",".join(map(str, vals))
        if not set(vals) <= {0, 1, 2}:
            row["reason"] = f"unexpected label values {vals}"; return row
        liver, tum = L == 1, L == 2
        row["liver_vox"], row["tumour_vox"] = int(liver.sum()), int(tum.sum())
        if not liver.any() or not tum.any():
            row["reason"] = "no liver or no tumour label"; return row
        sp = img.header.get_zooms()[:3]
        near = ndimage.distance_transform_edt(~liver, sampling=sp) <= 5.0           # swap guard: tumour must sit against the liver label
        row["tumour_near_liver"] = round(float(near[tum].mean()), 4); row["tumour_over_liver_vol"] = round(float(tum.sum() / liver.sum()), 4)
        ct = np.asarray(img.dataobj).astype(np.float32); row["liver_hu_ex_tumour"] = round(float(ct[liver].mean()), 1)
        if row["tumour_near_liver"] < 0.5 or row["tumour_over_liver_vol"] > 1.0:
            row["reason"] = "label inconsistency (tumour not against liver, or larger than liver)"; return row
        row["included"] = True
    except Exception as e:
        row["reason"] = f"error: {type(e).__name__}: {e}"
    return row


def contrast_one(name):
    img, lab = load(name); sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    ct = np.asarray(img.dataobj).astype(np.float32); L = np.asarray(lab.dataobj).astype(np.uint8)
    rows = case_lesions(ct, L == 2, L == 1, sp, n_lesions=None)
    nov = case_lesions(ct, L == 2, L == 1, sp, n_lesions=None)          # no vessel labels exist: identical by construction, kept so the column set matches CRML-CT
    return [{"patient": name.replace(".nii.gz", ""), "dataset": "msd_task03", **r, "contrast_hu_no_hepatic_vein": r["contrast_hu"], "contrast_hu_no_vessel_removal": b["contrast_hu"]} for r, b in zip(rows, nov)]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ds = json.load(open(os.path.join(root, "dataset.json")))
    names = sorted(os.path.basename(t["label"]) for t in ds["training"])
    if stage == "inventory":
        with Pool(10) as pool:
            rows = pool.map(inventory_one, names)
        write(os.path.join(outdir, "msd_inventory.csv"), rows)
        exc = [r for r in rows if not r["included"]]
        print(f"{len(rows)} cases, {len(rows) - len(exc)} included, {len(exc)} excluded")
        for r in exc:
            print("  EXCLUDED", r["case"], r["reason"])
    else:
        inv = list(csv.DictReader(open(os.path.join(outdir, "msd_inventory.csv"))))
        todo = [r["case"] + ".nii.gz" for r in inv if r["included"] == "True"]
        with Pool(10) as pool:
            rows = [x for rs in pool.map(contrast_one, todo) for x in rs]
        write(os.path.join(outdir, "msd_lesions.csv"), rows)
        print(len(rows), "lesions from", len({r["patient"] for r in rows}), "cases")
