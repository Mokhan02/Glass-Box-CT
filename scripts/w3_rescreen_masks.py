"""Predicted-liver masks of model A (swin, shipped pipeline up to `organ_post_process` for class 6, nearest-neighbour inversion as the shipped save_results) for an arbitrary list of converted HCC cases, for the 89-case translation re-screen (CPU analysis afterwards). No tumour output.
Usage (cwd = repo checkout, GPU): python scripts/w3_rescreen_masks.py <repo_A> <weights> <outdir> <converted root> <case> [<case> ...]"""
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Invertd

repo_a, wts, outdir, root = sys.argv[1:5]; cases = sys.argv[5:]
sys.path.insert(0, repo_a); sys.path.insert(0, ".")
from dataset.dataloader import get_loader_without_gt  # noqa: E402
from model.Universal_model import Universal_model  # noqa: E402
from utils.utils import NUM_CLASS, organ_post_process, threshold_organ  # noqa: E402

os.makedirs(os.path.join(outdir, "masks"), exist_ok=True); indir = os.path.join(outdir, "in"); os.makedirs(indir, exist_ok=True)
for c in cases:
    d = os.path.join(indir, f"{c}.nii.gz")
    if not os.path.lexists(d):
        os.symlink(os.path.join(root, c, "ct.nii.gz"), d)
args = types.SimpleNamespace(space_x=1.5, space_y=1.5, space_z=1.5, a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, data_root_path=indir, cache_dataset=False, cache_rate=0.6, log_name=os.path.join(outdir, "pp_log"))
loader, tf = get_loader_without_gt(args)
model = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone="swinunetr", encoding="word_embedding"); store = model.state_dict()
for k, v in torch.load(wts, map_location="cpu")["net"].items():
    store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
model.load_state_dict(store); model.cuda().eval(); torch.backends.cudnn.benchmark = True
for batch in loader:
    name = batch["name"][0]
    if name not in cases:
        continue
    with torch.no_grad():
        sig = F.sigmoid(sliding_window_inference(batch["image"].cuda(), (96, 96, 96), 1, model, overlap=0.5, mode="gaussian"))
    post = organ_post_process(threshold_organ(sig).cpu().numpy(), [6], os.path.join(outdir, "x", name), args)
    batch["k"] = torch.tensor(post[0, 5].astype(np.float64))[None, None]
    out = Invertd(keys=["k"], transform=tf, orig_keys="image", nearest_interp=True, to_tensor=True)(decollate_batch(batch)[0])["k"][0].numpy() > 0.5; del batch["k"]
    np.savez_compressed(os.path.join(outdir, "masks", f"swin__{name}.npz"), organ_full=out); print("done", name, int(out.sum()), flush=True)
print("RESCREEN_MASKS_DONE", flush=True)
