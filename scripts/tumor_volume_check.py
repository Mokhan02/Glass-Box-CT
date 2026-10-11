"""Does the HCC tumor-Dice gap between label routes (uid vs geometry) come from tumor size?
Usage: python scripts/tumor_volume_check.py cache/heldout_eval/hcc_tace_seg.csv"""
import csv
import json
import os
import sys

import numpy as np

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["target"] == "tumor"]
for r in rows:
    r["method"] = json.load(open(os.path.join("cache/heldout_nii/hcc_tace_seg", r["case"], "segments.json"))).get("label_method", "uid")
    r["dice"] = float(r["dice"]); r["gt"] = int(r["gt_voxels"])
for m in ("uid", "geometry"):
    rs = [r for r in rows if r["method"] == m]
    g = np.array([r["gt"] for r in rs]); d = np.array([r["dice"] for r in rs])
    print(f"{m:9s} n={len(rs):3d} GT tumor voxels: median {np.median(g):9.0f}  mean {g.mean():9.0f}  | Dice mean {d.mean():.3f}  frac Dice<0.1: {(d < 0.1).mean():.2f}")
g = np.array([r["gt"] for r in rows]); d = np.array([r["dice"] for r in rows])
q = np.quantile(g, [0, 1 / 3, 2 / 3, 1])
print("\nDice by GT tumor-size tercile (voxels in each case's own grid; grids differ between cases, so this is approximate):")
for i in range(3):
    m = (g >= q[i]) & (g <= q[i + 1])
    print(f"  tercile {i + 1} ({q[i]:.0f}..{q[i + 1]:.0f}) n={m.sum():3d} Dice {d[m].mean():.3f}  methods: " + ", ".join(f"{k}={sum(1 for r, mm in zip(rows, m) if mm and r['method'] == k)}" for k in ('uid', 'geometry')))
