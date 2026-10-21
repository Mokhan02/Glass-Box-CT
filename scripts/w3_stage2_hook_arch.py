"""W3 Stage 2, STEP 1, architecture-only check (CPU, no checkpoint, no data): builds model D's PlainConvUNet from the committed plan (results/w3_models/gate0/nnunet_task03_nnUNetPlans.json) with random weights, hooks every encoder stage, forwards a synthetic 128^3 patch and records channels, cumulative stride,
native token size in mm (plan spacing 1.0 x 0.768 x 0.768 mm, z y x) and the size after a 2x average pool. Shapes depend only on the architecture, not on weights. The real-case verification with the trained checkpoint is scripts/w3_stage2_hook_check.py (needs the instance).
Usage: python scripts/w3_stage2_hook_arch.py <plans json> <out json>"""
import json
import sys

import numpy as np
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet

plans, out = sys.argv[1:3]; P = json.load(open(plans)); c = P["configurations"]["3d_fullres"]; kw = dict(c["architecture"]["arch_kwargs"]); spacing = [float(s) for s in c["spacing"]]; ps = list(c["patch_size"])
for k in c["architecture"]["_kw_requires_import"]:
    if kw.get(k) is None:
        continue
    mod, name = kw[k].rsplit(".", 1); kw[k] = getattr(__import__(mod, fromlist=[name]), name)
net = PlainConvUNet(input_channels=1, num_classes=3, deep_supervision=False, **kw).eval(); enc = net.encoder; cum = np.cumprod(np.array(enc.strides), axis=0); caps = {}
for i, st in enumerate(enc.stages):
    st.register_forward_hook(lambda m, a, o, i=i: caps.__setitem__(i, tuple(o.shape)))
with torch.no_grad():
    y = net(torch.randn(1, 1, *ps))
R = {"mode": "architecture only (random weights, synthetic patch)", "network_class": type(net).__name__, "plan_spacing_mm_zyx": spacing, "patch_size": ps, "output_shape": list(y.shape), "stages": []}
for i, (ch, cs) in enumerate(zip(enc.output_channels, cum)):
    tok = [round(float(s * k), 1) for s, k in zip(spacing, cs)]
    R["stages"].append({"stage": i, "channels": int(ch), "stride_zyx": [int(v) for v in cs], "token_mm_zyx": tok, "token_mm_after_2x_pool_zyx": [round(2 * v, 1) for v in tok], "mean_inplane_token_mm": round((tok[1] + tok[2]) / 2, 1), "hook_output_shape": list(caps[i]),
                        "shape_ok": bool(caps[i] == (1, int(ch), *[p // int(k) for p, k in zip(ps, cs)]))})
R["all_shapes_ok"] = all(s["shape_ok"] for s in R["stages"]); json.dump(R, open(out, "w"), indent=1)
for s in R["stages"]:
    print(s["stage"], s["channels"], "ch | stride", s["stride_zyx"], "| token mm (z,y,x)", s["token_mm_zyx"], "| after 2x pool", s["token_mm_after_2x_pool_zyx"], "| hook", s["hook_output_shape"], "ok" if s["shape_ok"] else "BAD")
print("all shapes ok:", R["all_shapes_ok"], "| output", R["output_shape"])
