"""W3 Stage 1, model A (CLIP-Driven Universal Model; research use only) on the HCC UID cohort or the C4KC-KiTS kidney arm (results/w3_models/PREREG_stage1.md).
Inference replicates the authors' shipped `pred_pseudo.py` `validation` function for ONE tumour class (the equivalence with the full 32-class run is asserted on the first case): shipped loader and transforms, 96^3 Gaussian sliding window,
overlap 0.5, sigmoid, `threshold_organ`, `organ_post_process`, nearest-neighbour `Invertd`. Variants evaluated per lesion: raw thresholded sigmoid, (a) largest predicted-organ component, (a)+(b) closing, shipped (= (a)+(b)+(c) dilation).
Cases whose label the original converter displaced (results/dq_hcc_labels/corrected_cases.json: HCC_019_773068) are scored against the corrected label and contrast, and additionally against the as-converted ones under arms `asconv__*` (sensitivity row).
Alignment check on EVERY case (Amendment 2; against the corrected label for corrected cases): output shape equals the input shape, no axis flip of the predicted organ raises its Dice vs the GT organ, centroid offset <= 15 mm (array indices, GT spacing); the run stops if more than 2 cases fail; organ Dice < 0.85 is listed (sensitivity exclusion), not gating. The saved header affine of the shipped output is NOT used (stale).
Usage (cwd = repo checkout, PYTHONPATH not needed): python scripts/w3_modelA_run.py <repo_A> <weights> <swin|unet> <hcc|kidney> <outdir> [--limit N]"""
import csv
import glob
import json
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
from scipy import ndimage

repo_a, wts, bb, organ, outdir = sys.argv[1:6]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
sys.path.insert(0, repo_a); sys.path.insert(0, ".")
from dataset.dataloader import get_loader_without_gt  # noqa: E402
from model.Universal_model import Universal_model  # noqa: E402
from utils.utils import NUM_CLASS, ORGAN_NAME, TUMOR_ORGAN, merge_and_top_organ, organ_post_process, threshold_organ  # noqa: E402
from glassbox.seg import w3_lib as L  # noqa: E402

CLS = 27 if organ == "hcc" else 26                   # Liver Tumor / Kidney Tumor (1-based class index)
ORGANS = [6] if organ == "hcc" else [2, 3]           # predicted-organ channels used for the alignment check
ROOT = L.HCC_ROOT if organ == "hcc" else L.KITS_ROOT; SUF = "" if organ == "hcc" else "_kidney"
table = L.lesion_table(organ); table_asconv = L.lesion_table(organ, as_converted=True); corrected = L.corrected_cases() if organ == "hcc" else {}; cases = sorted({k[0] for k in table})
if limit:
    cases = cases[:limit]
os.makedirs(os.path.join(outdir, "masks"), exist_ok=True); indir = os.path.join(outdir, "in"); os.makedirs(indir, exist_ok=True)
for c in cases:
    dst = os.path.join(indir, f"{c}.nii.gz")
    if not os.path.lexists(dst):
        os.symlink(os.path.join(ROOT, c, "ct.nii.gz"), dst)
args = types.SimpleNamespace(space_x=1.5, space_y=1.5, space_z=1.5, a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, data_root_path=indir, cache_dataset=False, cache_rate=0.6, log_name=os.path.join(outdir, "pp_log"))
loader, val_transforms = get_loader_without_gt(args)
model = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone="swinunetr" if bb == "swin" else "unet", encoding="word_embedding"); store = model.state_dict()
for k, v in torch.load(wts, map_location="cpu")["net"].items():
    store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
model.load_state_dict(store); model.cuda().eval(); torch.backends.cudnn.benchmark = True


def invert(batch, mask, key):
    """Nearest-neighbour inversion to the input grid exactly as the shipped save_results (Invertd on the loader batch), without writing a file."""
    batch[key] = torch.tensor(mask.astype(np.float64))[None, None]
    out = Invertd(keys=[key], transform=val_transforms, orig_keys="image", nearest_interp=True, to_tensor=True)(decollate_batch(batch)[0])
    del batch[key]
    return out[key][0].numpy() > 0.5


rows, aligns, t0, failures, low_dice = [], [], time.time(), [], []; done = 0; extras = []
import nibabel as nib
nii_shape = {c: tuple(nib.load(os.path.join(L.case_root(organ, c, ROOT), c, 'ct.nii.gz')).shape) for c in cases}
for batch in loader:
    name = batch["name"][0]
    if name not in cases:
        continue
    t1 = time.time(); image = batch["image"].cuda()
    with torch.no_grad():
        sig = F.sigmoid(sliding_window_inference(image, (96, 96, 96), 1, model, overlap=0.5, mode="gaussian"))
    hard = threshold_organ(sig).cpu().numpy(); torch.cuda.empty_cache()
    olist = ORGANS + [CLS]; post = organ_post_process(hard.copy(), olist, os.path.join(outdir, "x", name), args)
    if done == 0:                                                          # equivalence of the restricted run with the shipped 32-class run (first case)
        full = organ_post_process(hard.copy(), list(range(1, 33)), os.path.join(outdir, "x", name), args)
        eq = all(np.array_equal(full[0, c - 1], post[0, c - 1]) for c in olist); extras.append({"restricted_equals_full_32_class_run_first_case": bool(eq)}); assert eq, "restricted post-processing differs from the full run"
    raw = hard[0, CLS - 1].astype(bool); organ_merge = merge_and_top_organ(hard[0], TUMOR_ORGAN[ORGAN_NAME[CLS - 1]]).astype(bool)
    closed = ndimage.binary_closing(organ_merge, structure=np.ones((5, 5, 5))); dil = ndimage.binary_dilation(closed, structure=np.ones((5, 5, 5)))
    step_a, step_ab, step_abc = raw & organ_merge, raw & closed, raw & dil; shipped = post[0, CLS - 1].astype(bool); steps_ok = bool(np.array_equal(shipped, step_abc))
    organ_pred = np.zeros(raw.shape, bool)
    for o in ORGANS:
        organ_pred |= post[0, o - 1].astype(bool)
    preds = {"raw": invert(batch, raw, "k_raw"), "step_a": invert(batch, step_a, "k_a"), "step_ab": invert(batch, step_ab, "k_ab"), "shipped": invert(batch, shipped, "k_ship")}
    organ_inv = invert(batch, organ_pred, "k_organ")
    arms = {f"A_{bb}_{k}{SUF}": v for k, v in preds.items()}; case_rows, geo = L.lesion_rows(organ, name, L.case_root(organ, name, ROOT), table, arms); rows += case_rows
    if name in corrected:          # sensitivity: the same predictions scored against the as-converted (displaced) label and contrast
        rows += L.lesion_rows(organ, name, ROOT, table_asconv, {f"asconv__{a}": v for a, v in arms.items()})[0]
    sh = tuple(geo["seg"].shape); al = L.alignment_report(organ_inv, geo["seg"], geo["ann"], geo["spacing"], nii_shape[name]); ad = al["organ_dice"]
    rec = {"case": name, **al, "input_grid_shape": list(organ_inv.shape), "ct_shape": list(sh), "shipped_equals_last_step": steps_ok,
           "voxels_preprocessed": {k: int(v.sum()) for k, v in {"raw": raw, "step_a": step_a, "step_ab": step_ab, "shipped": shipped}.items()}, "seconds": round(time.time() - t1, 1)}
    aligns.append(rec)
    if not rec["pass"]:
        failures.append(name)
    if rec["low_dice_lt_0.85"]:
        low_dice.append(name)
    np.savez_compressed(os.path.join(outdir, "masks", f"{bb}__{name}.npz"), ann=geo["ann"], organ_full=organ_inv[:sh[0], :sh[1], :sh[2]], **{k: v[:sh[0], :sh[1], :sh[2]][:, :, geo["ann"]] for k, v in preds.items()})      # organ_full: the shipped predicted organ on the whole input grid (added for the shift-test gate)
    done += 1; print(f"[{bb} {organ}] {done}/{len(cases)} {name} organ_dice {ad:.3f} offset {rec['centroid_offset_mm']} mm {'PASS' if rec['pass'] else 'FAIL'} {rec['seconds']}s elapsed {time.time() - t0:.0f}s", flush=True)
    if len(failures) > L.ALIGN_MAX_FAILURES:
        print(f"STOP: alignment failed on {len(failures)} cases (> {L.ALIGN_MAX_FAILURES}): {failures}", flush=True); break
if rows:
    with open(os.path.join(outdir, f"w3_lesions_A_{bb}_{organ}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=L.FIELDS); w.writeheader(); w.writerows(rows)
json.dump({"backbone": bb, "organ": organ, "cases_run": done, "cases_expected": len(cases), "gate": "Amendment 2: shape equals input, no axis flip raises Dice, centroid offset <= %.0f mm" % L.ALIGN_MAX_OFFSET_MM, "n_failures": len(failures), "failures": failures, "low_dice_lt_0.85": low_dice, "stopped_early": len(failures) > L.ALIGN_MAX_FAILURES,
           "extras": extras, "per_case": aligns, "torch": torch.__version__}, open(os.path.join(outdir, f"w3_alignment_A_{bb}_{organ}.json"), "w"), indent=1)
print(f"DONE {bb} {organ}: {done} cases, {len(rows)} rows, {len(failures)} alignment failures, {time.time() - t0:.0f}s", flush=True)
