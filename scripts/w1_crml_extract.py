"""W1, CRML-CT. Two stages (results/w1_training_audit/PREREG.md and DEVIATIONS.md #1-#2).
  inventory  parse every patient, apply the exclusion criteria, cache CT/liver/tumour arrays, write crml_inventory.csv (per-patient log, NO contrast computed).
  contrast   read the inventory, compute lesion-level contrast for the included patients, write crml_lesions.csv.
Usage: python -I scripts/w1_crml_extract.py inventory|contrast data/w1/crlm data/w1/cache_crml results/w1_training_audit"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
import pydicom

sys.path.insert(0, ".")
from glassbox.audit.contrast import case_lesions  # noqa: E402
from glassbox.audit.crml import choose_seg_series, load_ct, load_seg_masks, roles  # noqa: E402

stage, root, cache, outdir = sys.argv[1:5]


def inventory_one(p):
    row = {"patient": p, "included": False, "reason": "", "series_used": "", "series_seen": "", "n_tumour_segments": 0, "tumour_labels": "", "liver_vox": 0, "tumour_vox": 0, "tumour_in_liver": np.nan,
           "tumour_over_liver_vol": np.nan, "liver_hu_ex_tumour": np.nan, "shape": "", "spacing": "", "max_gap_dev_mm": np.nan}
    try:
        pd_ = os.path.join(root, p)
        seg, info = choose_seg_series(pd_); row["series_seen"] = info
        if seg is None:
            row["reason"] = "no single manual series: " + info; return row
        row["series_used"] = os.path.basename(os.path.dirname(seg))
        ct, sp, uids, gdev, pos = load_ct(pd_); row.update(shape="x".join(map(str, ct.shape)), spacing="x".join(f"{s:.3f}" for s in sp), max_gap_dev_mm=round(gdev, 3))
        d = pydicom.dcmread(seg, stop_before_pixels=True); r = roles(d)
        labs = {int(s.SegmentNumber): str(s.SegmentLabel) for s in d.SegmentSequence}
        livers = [k for k, v in r.items() if v == "liver"]; tums = [k for k, v in r.items() if v == "tumour"]
        row["n_tumour_segments"] = len(tums); row["tumour_labels"] = ",".join(labs[k] for k in tums)
        if len(livers) != 1 or not tums:
            row["reason"] = f"{len(livers)} liver segments, {len(tums)} tumour segments"; return row
        m = load_seg_masks(seg, uids, ct.shape)
        liver = m[livers[0]]; tum = np.zeros(ct.shape, bool)
        for k in tums:
            tum |= m[k]
        row.update(liver_vox=int(liver.sum()), tumour_vox=int(tum.sum()))
        if gdev > 1.0:
            row["reason"] = f"slice gap deviation {gdev:.2f} mm > 1 mm"; return row
        if not liver.any() or not tum.any():
            row["reason"] = "empty liver or tumour mask"; return row
        row["tumour_in_liver"] = round(float((tum & liver).sum() / tum.sum()), 4); row["tumour_over_liver_vol"] = round(float(tum.sum() / liver.sum()), 4)
        row["liver_hu_ex_tumour"] = round(float(ct[liver & ~tum].astype(np.float32).mean()), 1) if (liver & ~tum).any() else np.nan
        if row["tumour_in_liver"] < 0.5 or row["tumour_over_liver_vol"] > 1.0:
            row["reason"] = "label inconsistency (tumour not inside liver, or larger than liver)"; return row
        np.savez_compressed(os.path.join(cache, p + ".npz"), ct=ct, liver=liver, tumour=tum, spacing=np.array(sp))
        row["included"] = True
    except Exception as e:
        row["reason"] = f"error: {type(e).__name__}: {e}"
    return row


def contrast_one(p):
    z = np.load(os.path.join(cache, p + ".npz")); sp = tuple(float(x) for x in z["spacing"])
    rows = case_lesions(z["ct"].astype(np.float32), z["tumour"], z["liver"], sp, n_lesions=None)       # all lesions >= 10 mm, ranked; the 2 largest are rank 1-2
    return [{"patient": p, "dataset": "crml", **r} for r in rows]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    os.makedirs(cache, exist_ok=True)
    if stage == "inventory":
        pats = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        with Pool(6) as pool:
            rows = pool.map(inventory_one, pats)
        write(os.path.join(outdir, "crml_inventory.csv"), rows)
        exc = [r for r in rows if not r["included"]]
        print(f"{len(rows)} patients, {len(rows) - len(exc)} included, {len(exc)} excluded")
        for r in exc:
            print("  EXCLUDED", r["patient"], r["reason"])
    else:
        inv = list(csv.DictReader(open(os.path.join(outdir, "crml_inventory.csv"))))
        pats = [r["patient"] for r in inv if r["included"] == "True"]
        with Pool(6) as pool:
            rows = [x for rs in pool.map(contrast_one, pats) for x in rs]
        write(os.path.join(outdir, "crml_lesions.csv"), rows)
        print(len(rows), "lesions from", len({r["patient"] for r in rows}), "patients")
