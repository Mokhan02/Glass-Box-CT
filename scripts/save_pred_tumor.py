"""Save VISTA3D's predicted hepatic-tumor mask (class 26, p > 0.5, original CT geometry) for each converted HCC case.
Usage: python scripts/save_pred_tumor.py --out cache/heldout_pred/hcc_tace_seg [--limit N]"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from glassbox.seg.infer import load_network, predict_probs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="cache/heldout_nii/hcc_tace_seg")
ap.add_argument("--out", default="cache/heldout_pred/hcc_tace_seg")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda")
net = load_network("cache/weights/model.pt", dev)
cases = sorted(c for c in os.listdir(a.root) if os.path.exists(os.path.join(a.root, c, ".done")))
for i, c in enumerate(cases[: a.limit or None], 1):
    dst = os.path.join(a.out, f"{c}.npz")
    if os.path.exists(dst):
        continue
    probs, _ = predict_probs(os.path.join(a.root, c, "ct.nii.gz"), [26], net, dev)
    np.savez_compressed(dst + ".part.npz", pred=(probs[0].numpy() > 0.5))
    os.replace(dst + ".part.npz", dst)
    print(f"[{i}/{len(cases)}] {c}", flush=True)
