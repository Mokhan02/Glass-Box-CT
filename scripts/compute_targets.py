"""Per-scan RQ1 targets and baseline features from CT + labels (TotalSegmentator layout under --root).
Writes <out>/targets.csv and <out>/baselines.npz. Scans without L3 coverage are listed with empty muscle fields."""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from glassbox.bodycomp.muscle import PARASPINAL, PSOAS, global_hu_histogram, l3_muscle, load_canonical  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="cache/ts")
ap.add_argument("--out", default="cache/targets")
ap.add_argument("--ids", nargs="*")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
meta = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(a.root, "meta.csv"), encoding="utf-8-sig"), delimiter=";")}
ids = a.ids or sorted(d for d in os.listdir(a.root) if os.path.exists(os.path.join(a.root, d, ".done")))
rows, hist, local = [], {}, {}
for sid in ids:
    ct, zooms = load_canonical(os.path.join(a.root, sid, "ct.nii.gz"))
    masks = {n: load_canonical(os.path.join(a.root, sid, "segmentations", f"{n}.nii.gz"))[0]
             for n in ("vertebrae_L3",) + PSOAS + PARASPINAL}
    m = l3_muscle(ct, masks, zooms)
    hist[sid] = global_hu_histogram(ct)
    row = {"image_id": sid, **{k: meta[sid][k] for k in ("age", "gender", "kvp", "manufacturer", "scanner_model", "split")}}
    if m:
        local[sid] = m.pop("local_hu")
        row.update(m)
    rows.append(row)
    print(sid, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k.startswith(("muscle", "psoas", "lama", "l3"))}, flush=True)
cols = ["image_id", "age", "gender", "kvp", "manufacturer", "scanner_model", "split", "l3_slice", "muscle_area_cm2",
        "psoas_area_cm2", "muscle_mean_hu", "muscle_sd_hu", "lama_fraction"]
with open(os.path.join(a.out, "targets.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
np.savez(os.path.join(a.out, "baselines.npz"), hist_ids=list(hist), hist=np.stack(list(hist.values())),
         local_ids=list(local), local=np.stack(list(local.values())) if local else np.zeros((0, 7)))
