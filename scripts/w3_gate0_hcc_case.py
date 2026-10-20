"""W3 Gate 0, model A on ONE HCC case (research use only; the shipped repo code is run from the pinned commit under PYTHONPATH, not from inside the clone).
Checks: (1) the shipped `pred_pseudo.py` 'Liver Tumor' output is non-empty, on the CT grid, and overlaps the ground-truth lesions; (2) hidden-state shapes of `backbone.swinViT` on every real sliding window (expected output[2] = (1, 192, 12, 12, 12));
(3) running the shipped `organ_post_process` for class 27 only gives the same class-27 mask as running it for all 32 classes (so the restricted run used for the cohort is equivalent); (4) the cumulative post-processing steps on this case.
Usage: python scripts/w3_gate0_hcc_case.py <repo_A dir> <weights> <input dir with the case .nii.gz> <shipped out dir> <case> <gt seg nii.gz> <out json>"""
import json
import sys
import types

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from scipy import ndimage

repo, wts, indir, shipped, case, gtp, out = sys.argv[1:8]
sys.path.insert(0, repo)
from dataset.dataloader import get_loader_without_gt  # noqa: E402
from model.Universal_model import Universal_model  # noqa: E402
from utils.utils import NUM_CLASS, ORGAN_NAME, extract_topk_largest_candidates, organ_post_process, threshold_organ  # noqa: E402

R = {"case": case}
# (1) shipped output
ct = nib.load(f"{indir}/{case}.nii.gz"); tum = nib.load(f"{shipped}/{case}/{case}_Liver Tumor.nii.gz"); t = np.asarray(tum.dataobj) > 0
R["shipped_liver_tumor"] = {"shape_equals_ct": tuple(tum.shape) == tuple(ct.shape), "affine_close": bool(np.allclose(tum.affine, ct.affine, atol=1e-3)), "voxels": int(t.sum()), "mL": round(float(t.sum()) * float(np.prod(ct.header.get_zooms()[:3])) / 1000, 1), "dtype": str(tum.get_data_dtype())}
seg = np.asarray(nib.load(gtp).dataobj); ann = (seg > 0).any(axis=(0, 1)); gt = (seg == 2); lab, n = ndimage.label(gt[:, :, ann], structure=np.ones((3, 3, 3)))
R["gt_lesions_on_annotated_slices"] = [{"voxels": int((lab == i).sum()), "coverage_by_shipped_output": round(float((t[:, :, ann] & (lab == i)).sum() / (lab == i).sum()), 3)} for i in range(1, n + 1)]
# (2) hook shapes on real windows, (3) post-processing equivalence, (4) cumulative steps
args = types.SimpleNamespace(space_x=1.5, space_y=1.5, space_z=1.5, a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, data_root_path=indir, cache_dataset=False, cache_rate=0.6)
loader, _ = get_loader_without_gt(args); batch = next(iter(loader)); image = batch["image"].cuda()
model = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone="swinunetr", encoding="word_embedding"); store = model.state_dict(); ld = torch.load(wts, map_location="cpu")["net"]
for k, v in ld.items():
    store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
model.load_state_dict(store); model.cuda().eval(); shapes = []
model.backbone.swinViT.register_forward_hook(lambda m, a, o: shapes.append([tuple(x.shape) for x in o]))
with torch.no_grad():
    pred = sliding_window_inference(image, (96, 96, 96), 1, model, overlap=0.5, mode="gaussian"); sig = F.sigmoid(pred)
hard = threshold_organ(sig).cpu().numpy()
R["preprocessed_image_shape"] = list(image.shape); R["windows_run"] = len(shapes); R["output2_shape_all_windows_equal"] = bool(all(s[2] == (1, 192, 12, 12, 12) for s in shapes)); R["output2_shape"] = list(shapes[0][2]); R["all_five_shapes_first_window"] = [list(x) for x in shapes[0]]
args2 = types.SimpleNamespace(log_name="/tmp/w3_gate0_log")
full = organ_post_process(hard.copy(), list(range(1, 33)), "x/y/z", args2); only27 = organ_post_process(hard.copy(), [27], "x/y/z", args2)
R["postproc_class27_only_equals_full_list"] = bool(np.array_equal(full[0, 26], only27[0, 26]))
raw27 = hard[0, 26].astype(bool); liver_raw = hard[0, 5].astype(bool); liver_top1 = extract_topk_largest_candidates(liver_raw.astype(np.uint8), 1).astype(bool)
closed = ndimage.binary_closing(liver_top1, structure=np.ones((5, 5, 5))); dil = ndimage.binary_dilation(closed, structure=np.ones((5, 5, 5)))
steps = {"raw_threshold": raw27, "intersect_raw_predicted_liver": raw27 & liver_raw, "intersect_largest_liver_component": raw27 & liver_top1, "intersect_closed_liver": raw27 & closed, "shipped_closed_and_dilated_liver": raw27 & dil}
R["cumulative_steps_voxels_preprocessed_grid"] = {k: int(v.sum()) for k, v in steps.items()}; R["shipped_postprocessed_voxels_preprocessed_grid"] = int(full[0, 26].sum()); R["shipped_equals_last_step"] = bool(np.array_equal(full[0, 26].astype(bool), steps["shipped_closed_and_dilated_liver"]))
json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1))
