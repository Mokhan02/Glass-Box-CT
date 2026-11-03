"""W4 (WAW-TACE) Universal Model (CLIP-Driven; Swin-UNETR backbone; research use only) predictions: the SHIPPED variant of W3 Stage 1 (raw threshold, largest-organ component, closing, dilation = `pred_pseudo.py` post-processing for the liver-tumour class 27),
the same inference core as scripts/w3_modelA_run.py (shipped loader and transforms, 96^3 Gaussian sliding window, overlap 0.5, sigmoid, nearest-neighbour Invertd), on every series of the listed patients in a ground-truth root.
Writes <out>/<pid>_<series>.npz (bool arrays `tumour` = shipped tumour, `liver` = predicted liver organ, both inverted to the CT grid); existing files are skipped. Equivalence of the restricted post-processing with the full 32-class run is asserted on the first processed series.
Usage (cwd = repo checkout, venv_a python): python scripts/w4_waw_predict_A.py <repo_A> <weights> <gt root> <patients file> <out dir> [--limit N]"""
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Invertd

repo_a, wts, gt, pfile, out = sys.argv[1:6]; limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
sys.path.insert(0, repo_a); sys.path.insert(0, ".")
from dataset.dataloader import get_loader_without_gt  # noqa: E402
from model.Universal_model import Universal_model  # noqa: E402
from utils.utils import NUM_CLASS, organ_post_process, threshold_organ  # noqa: E402

CLS, ORGANS = 27, [6]
pats = set(open(pfile).read().split()); cases = sorted((d for d in os.listdir(gt) if d.split("_")[0] in pats), key=lambda d: (int(d.split("_")[0]), d))[:limit]; os.makedirs(out, exist_ok=True)
todo = [c for c in cases if not os.path.exists(os.path.join(out, c + ".npz"))]; indir = os.path.join(out, "_in"); os.makedirs(indir, exist_ok=True)
for c in todo:
    dst = os.path.join(indir, f"{c}.nii.gz")
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(os.path.join(gt, c, "ct.nii.gz")), dst)
args = types.SimpleNamespace(space_x=1.5, space_y=1.5, space_z=1.5, a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, data_root_path=indir, cache_dataset=False, cache_rate=0.6, log_name=os.path.join(out, "pp_log"))
loader, val_transforms = get_loader_without_gt(args)
model = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone="swinunetr", encoding="word_embedding"); store = model.state_dict()
for k, v in torch.load(wts, map_location="cpu")["net"].items():
    store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
model.load_state_dict(store); model.cuda().eval(); torch.backends.cudnn.benchmark = True


def invert(batch, mask, key):
    batch[key] = torch.tensor(mask.astype(np.float64))[None, None]
    o = Invertd(keys=[key], transform=val_transforms, orig_keys="image", nearest_interp=True, to_tensor=True)(decollate_batch(batch)[0]); del batch[key]
    return o[key][0].numpy() > 0.5


t0 = time.time(); done = 0
for batch in loader:
    name = batch["name"][0]
    if name not in todo:
        continue
    with torch.no_grad():
        sig = F.sigmoid(sliding_window_inference(batch["image"].cuda(), (96, 96, 96), 1, model, overlap=0.5, mode="gaussian"))
    hard = threshold_organ(sig).cpu().numpy(); torch.cuda.empty_cache(); post = organ_post_process(hard.copy(), ORGANS + [CLS], os.path.join(out, "x", name), args)
    if done == 0:
        full = organ_post_process(hard.copy(), list(range(1, 33)), os.path.join(out, "x", name), args); assert all(np.array_equal(full[0, c - 1], post[0, c - 1]) for c in ORGANS + [CLS]), "restricted post-processing differs from the full run"
    tum = invert(batch, post[0, CLS - 1].astype(bool), "k_ship"); liv = invert(batch, post[0, ORGANS[0] - 1].astype(bool), "k_organ")
    np.savez_compressed(os.path.join(out, name + ".tmp.npz"), tumour=tum, liver=liv); os.replace(os.path.join(out, name + ".tmp.npz"), os.path.join(out, name + ".npz")); done += 1
    print(f"[A w4] {done}/{len(todo)} {name} elapsed {time.time() - t0:.0f}s", flush=True)
print(f"DONE A w4: {done} new series of {len(cases)}, {time.time() - t0:.0f}s", flush=True)
