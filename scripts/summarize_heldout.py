"""Per-case table of held-out results with how each label volume was built (uid vs geometry).
Usage: python scripts/summarize_heldout.py cache/heldout_eval_test/hcc_tace_seg.csv hcc_tace_seg [cache/heldout_nii]"""
import csv
import json
import os
import sys

import numpy as np

path, coll = sys.argv[1], sys.argv[2]
root = sys.argv[3] if len(sys.argv) > 3 else "cache/heldout_nii"
rows = list(csv.DictReader(open(path)))
by = {}
for r in rows:
    meta = json.load(open(os.path.join(root, coll, r["case"], "segments.json")))
    r["method"] = meta.get("label_method", "uid")
    r["dice_f"] = float(r["dice"]); r["full"] = float(r["dice_full"]); r["ratio"] = int(r["pred_voxels"]) / max(int(r["gt_voxels"]), 1)
    by.setdefault((r["target"], r["method"]), []).append(r)
for r in rows:
    print(f"{r['case'][:15]:16s} {r['method']:9s} {r['target']:6s} dice {r['dice_f']:.3f} (full {r['full']:.3f}, slab {float(r['slab_fraction']):.2f})  gt {int(r['gt_voxels']):8d}  pred {int(r['pred_voxels']):8d}  pred/gt {r['ratio']:5.2f}")
print("\nmean Dice by target and label method:")
for (t, m), rs in sorted(by.items()):
    print(f"  {t:7s} {m:9s} n={len(rs):3d}  dice {np.mean([x['dice_f'] for x in rs]):.3f} (full {np.mean([x['full'] for x in rs]):.3f})  median pred/gt {np.median([x['ratio'] for x in rs]):.2f}")
