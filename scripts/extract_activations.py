"""Extract activation tokens for TotalSegmentator scans already fetched into --root (scripts/fetch_ts.py).
Usage (repo root): python scripts/extract_activations.py --root cache/ts --out cache/acts --device cuda [--ids s0235 ...]
Resumable: a scan whose output .npz exists is skipped."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from monai.inferers import SlidingWindowInfererAdapt  # noqa: E402
from monai.transforms import LoadImage  # noqa: E402

from glassbox.data.preprocess import PATCH_SIZE, SW_OVERLAP, build_pair_transforms  # noqa: E402
from glassbox.extract.hooks import ActivationRecorder  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

LABELS = ["spleen", "liver", "aorta", "iliopsoas_left", "iliopsoas_right",
          "autochthon_left", "autochthon_right", "vertebrae_L3"]  # label id = index + 1, 0 = background

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="cache/ts")
ap.add_argument("--out", default="cache/acts")
ap.add_argument("--device", default="cuda")
ap.add_argument("--ids", nargs="*")
ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                help="used when --ids is not given: only scans in this TotalSegmentator split (from meta.csv)")
ap.add_argument("--weights", default="cache/weights/model.pt")
ap.add_argument("--random-init", action="store_true", help="baseline: same architecture, untrained weights (seed 0)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device(a.device)
if a.random_init:
    from monai.networks.nets import vista3d132

    torch.manual_seed(0)
    net = vista3d132(encoder_embed_dim=48, in_channels=1).to(dev).eval()
else:
    net = load_network(a.weights, dev)
import csv  # noqa: E402

if a.ids:
    ids = a.ids
else:
    split = {r["image_id"]: r["split"] for r in csv.DictReader(open(os.path.join(a.root, "meta.csv"), encoding="utf-8-sig"), delimiter=";")}
    ids = sorted(d for d in os.listdir(a.root)
                 if os.path.isdir(os.path.join(a.root, d)) and split.get(d) == a.split
                 and os.path.exists(os.path.join(a.root, d, ".done")))  # fully fetched only
print(f"{len(ids)} scans, split={a.split if not a.ids else 'explicit'}", flush=True)
load = LoadImage(image_only=True)
for n, sid in enumerate(ids, 1):
    dst = os.path.join(a.out, f"{sid}.npz")
    if os.path.exists(dst):
        continue
    t = time.time()
    img = load(os.path.join(a.root, sid, "ct.nii.gz"))
    lab = torch.zeros_like(img, dtype=torch.int16)
    for i, c in enumerate(LABELS, 1):
        lab[load(os.path.join(a.root, sid, "segmentations", f"{c}.nii.gz")) > 0.5] = i
    from monai.data import MetaTensor
    lab = MetaTensor(lab, meta=img.meta.copy(), affine=img.affine)
    d = build_pair_transforms()({"image": img, "label": lab})
    x = d["image"].unsqueeze(0).to(dev)  # [1,1,H,W,D]
    rec = ActivationRecorder(net, d["label"][0].as_tensor().long().to("cpu"), len(LABELS), seed=int(sid[1:]))
    rec.label = rec.label.to(dev)
    sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP,
                                   with_coord=True, padding_mode="replicate")
    with torch.no_grad():
        sw(x, net, transpose=True, point_coords=None, point_labels=None,
           class_vector=torch.tensor([[1]], device=dev), prompt_class=None, prev_mask=None, labels=None, label_set=None)
    toks = rec.collect()
    rec.close()
    with open(dst + ".part", "wb") as f:  # atomic: a killed run never leaves a half-written .npz behind
            np.savez(f, shape=np.array(x.shape[2:]), n_windows=len(rec.coords), labels=np.array(["background"] + LABELS), **toks)
    os.replace(dst + ".part", dst)
    print(f"[{n}/{len(ids)}] {sid} windows={len(rec.coords)} "
          + " ".join(f"{k}={len(toks[k + '_label'])}" for k in ("bottleneck", "mid", "decoder_F"))
          + f" {time.time() - t:.0f}s", flush=True)
