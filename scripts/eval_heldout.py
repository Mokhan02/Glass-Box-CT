"""Held-out Dice for VISTA3D on converted external-test cases (scripts/convert_heldout.py output).
Targets (ground-truth segments -> model class prompt(s); unions where several):
  c4kc_kits     kidney (seg 1 'Kidney' + seg 2 'Mass') <- right kidney 5 + left kidney 14.  The model's organ includes the lesion
                inside it, so the GT organ must too. The mass alone is skipped: kidney mass (129) needs point prompts.
  hcc_tace_seg  liver (seg 1 + seg 2 'Mass') <- 1;  tumor (seg 2) <- hepatic tumor 26.
                Aorta and portal vein are NOT evaluated: their GT is a short fragment, not the whole vessel.
Dice is computed only on slices that carry annotation (any nonzero GT label on the slice). The HCC SEGs annotate part of the scan,
sometimes every other slice, so unlabeled slices would count correct predictions as false positives. Limitation: a slice where
the model predicts the organ but the GT is empty is excluded too, so false positives there are not penalised.
`dice_full` is the unrestricted value, kept for transparency; `slab_fraction` is the share of slices scored.
Writes one row per (case, target) with Dice plus the confidence signals RQ2 compares against:
mean sigmoid probability and mean binary entropy inside the predicted mask, and predicted volume.
Usage: python scripts/eval_heldout.py --collection c4kc_kits --out cache/heldout_eval --device cuda [--limit N]"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from monai.transforms import LoadImage  # noqa: E402

from glassbox.data.results import ResultsTable  # noqa: E402
from glassbox.seg.infer import dice, load_network, predict_probs  # noqa: E402

TARGETS = {
    "c4kc_kits": {"kidney": ((1, 2), (5, 14))},
    "hcc_tace_seg": {"liver": ((1, 2), (1,)), "tumor": ((2,), (26,))},
}
ap = argparse.ArgumentParser()
ap.add_argument("--collection", required=True, choices=list(TARGETS))
ap.add_argument("--root", default="cache/heldout_nii")
ap.add_argument("--out", default="cache/heldout_eval")
ap.add_argument("--device", default="cuda")
ap.add_argument("--weights", default="cache/weights/model.pt")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device(a.device)
net = load_network(a.weights, dev)
load = LoadImage(image_only=True)
base = os.path.join(a.root, a.collection)
cases = sorted(c for c in os.listdir(base) if os.path.exists(os.path.join(base, c, ".done")))
if a.limit:
    cases = cases[: a.limit]
classes = sorted({c for _, cs in TARGETS[a.collection].values() for c in cs})
table = ResultsTable(["case", "target", "dice", "dice_full", "slab_fraction", "gt_voxels", "pred_voxels", "mean_prob_in_pred", "mean_entropy_in_pred"])
path = os.path.join(a.out, f"{a.collection}.csv")
for n, case in enumerate(cases, 1):
    t = time.time()
    try:
        probs, _ = predict_probs(os.path.join(base, case, "ct.nii.gz"), classes, net, dev)
        seg = load(os.path.join(base, case, "seg.nii.gz")).numpy()
        probs = probs.numpy()
        if probs.shape[1:] != seg.shape:
            raise RuntimeError(f"shape mismatch {probs.shape[1:]} vs {seg.shape}")
        ann = (seg > 0).any(axis=(0, 1))  # slices carrying annotation
        for target, (seg_ids, cls) in TARGETS[a.collection].items():
            gt_full = np.isin(seg, seg_ids)
            if gt_full.sum() == 0:
                continue
            p_full = np.max([probs[classes.index(c)] for c in cls], axis=0)  # union of prompts
            gt, p = gt_full[:, :, ann], p_full[:, :, ann]
            pred = p > 0.5
            pin = p[pred]
            ent = -(pin * np.log(pin + 1e-8) + (1 - pin) * np.log(1 - pin + 1e-8)) if pred.any() else np.array([np.nan])
            table.add(a.collection, 1, case=case, target=target, dice=round(dice(pred, gt), 4), dice_full=round(dice(p_full > 0.5, gt_full), 4),
                      slab_fraction=round(float(ann.mean()), 3), gt_voxels=int(gt.sum()),
                      pred_voxels=int(pred.sum()), mean_prob_in_pred=round(float(pin.mean()), 4) if pred.any() else float("nan"),
                      mean_entropy_in_pred=round(float(np.nanmean(ent)), 4))
        last = [r for r in table.rows if r["case"] == case]
        print(f"[{n}/{len(cases)}] {case} " + " ".join(f"{r['target']}={r['dice']:.3f}" for r in last) + f" {time.time() - t:.0f}s", flush=True)
    except Exception as e:
        print(f"[{n}/{len(cases)}] FAILED {case}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    table.to_csv(path)  # rewrite as we go so a stopped run keeps its results
print("wrote", path, "rows", len(table.rows))
