"""W3 Stage 2, STEP 1 (CPU only): model D's hook point. Loads the fold-0 nnU-Net (checkpoint_final) on the CPU, lists every encoder stage (channels, cumulative stride, native token size in mm from the plan spacing, token size after a 2x average pool as the VISTA-3D probe uses),
registers a forward hook on every encoder stage, preprocesses ONE real HCC case exactly as nnU-Net does (DefaultPreprocessor: crop, CT normalisation, resample), forwards a 128^3 patch centred on the lesion and records the hook output shapes.
Usage (cwd = repo checkout, nnunet env): python scripts/w3_stage2_hook_check.py <model folder> <ct.nii.gz> <seg.nii.gz> <out json>"""
import json
import sys

import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

mf, ct_path, seg_path, out = sys.argv[1:5]
pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=False, device=torch.device("cpu"), verbose=False)
pred.initialize_from_trained_model_folder(mf, use_folds=(0,), checkpoint_name="checkpoint_final.pth"); net = pred.network.eval(); cm = pred.configuration_manager
enc = net.encoder; spacing = [float(s) for s in cm.spacing]; strides = [list(s) if isinstance(s, (list, tuple)) else [s] * 3 for s in enc.strides]; cum = np.cumprod(np.array(strides), axis=0); chans = list(enc.output_channels)
R = {"network_class": type(net).__name__, "compiled_wrapper": type(pred.network).__name__, "plan_spacing_mm_zyx": spacing, "patch_size": list(cm.patch_size), "stages": []}
for i, (c, cs) in enumerate(zip(chans, cum)):
    tok = [round(float(s * k), 1) for s, k in zip(spacing, cs)]
    R["stages"].append({"stage": i, "channels": int(c), "stride_zyx": [int(v) for v in cs], "token_mm_zyx": tok, "token_mm_after_2x_pool_zyx": [round(2 * v, 1) for v in tok], "mean_inplane_token_mm": round((tok[1] + tok[2]) / 2, 1)})
caps = {}
for i, st in enumerate(enc.stages):
    st.register_forward_hook(lambda m, a, o, i=i: caps.__setitem__(i, tuple(o.shape)))
data, _, props = cm.preprocessor_class(verbose=False).run_case([ct_path], None, pred.plans_manager, cm, pred.dataset_json)
seg = np.asarray(nib.load(seg_path).dataobj); idx = np.argwhere(seg == 2); frac = idx.mean(0) / np.array(seg.shape); R["preprocessed_shape_czyx"] = list(data.shape)
z = [int(round(f * n)) for f, n in zip(frac[::-1], data.shape[1:])]; ps = list(cm.patch_size); lo = [max(0, min(zc - p // 2, n - p)) for zc, p, n in zip(z, ps, data.shape[1:])]
patch = torch.from_numpy(np.ascontiguousarray(data[:, lo[0]:lo[0] + ps[0], lo[1]:lo[1] + ps[1], lo[2]:lo[2] + ps[2]])).float()[None]
R["patch_shape"] = list(patch.shape)
with torch.no_grad():
    y = net(patch)
y = y[0] if isinstance(y, (list, tuple)) else y; R["output_shape"] = list(y.shape)
for s in R["stages"]:
    s["hook_output_shape"] = list(caps[s["stage"]]); s["shape_ok"] = bool(caps[s["stage"]] == (1, s["channels"], *[p // k for p, k in zip(ps, s["stride_zyx"])]))
R["all_shapes_ok"] = all(s["shape_ok"] for s in R["stages"]); json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1))
