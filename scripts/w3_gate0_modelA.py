"""W3 Gate 0, model A (CLIP-Driven Universal Model; research use only, non-commercial; never published or merged into the product branch): load the two released checkpoints into the repo's own model class exactly as its `pred_pseudo.py` does,
verify tensor shapes, the class table (26 Kidney Tumor, 27 Liver Tumor -> output channels 25 and 26), the class-embedding source (checkpoint buffer vs the repo's `txt_encoding.pth`), and the Swin encoder hook shapes on a dummy 96^3 window.
The model code is read from a separate clone directory passed on the command line (never executed from inside it); the repo's `utils` is parsed with `ast`, not imported.
Usage: python scripts/w3_gate0_modelA.py <repo clone dir> <weights dir> <out json>"""
import ast
import json
import sys

import numpy as np
import torch

repo, wdir, out = sys.argv[1:4]
sys.path.insert(0, repo)
from model.Universal_model import Universal_model  # noqa: E402

src = open(f"{repo}/utils/utils.py").read(); tree = ast.parse(src); consts = {}
for n in tree.body:
    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("NUM_CLASS", "ORGAN_NAME", "TUMOR_ORGAN", "THRESHOLD_DIC", "TUMOR_NUM", "TUMOR_SIZE"):
        try:
            consts[n.targets[0].id] = ast.literal_eval(n.value)
        except Exception:
            pass
NUM_CLASS, ORGAN = consts["NUM_CLASS"], consts["ORGAN_NAME"]; R = {"num_class": NUM_CLASS, "n_organ_names": len(ORGAN)}
R["class_table"] = {"26": ORGAN[25], "27": ORGAN[26]}; R["tumor_organ_merge"] = {k: consts["TUMOR_ORGAN"][k] for k in ("Kidney Tumor", "Liver Tumor")}; R["threshold"] = {k: consts["THRESHOLD_DIC"][k] for k in ("Kidney Tumor", "Liver Tumor")}
R["tumor_num_size_postproc"] = {"TUMOR_NUM": {k: consts["TUMOR_NUM"][k] for k in ("Kidney Tumor", "Liver Tumor")}, "TUMOR_SIZE": {k: consts["TUMOR_SIZE"][k] for k in ("Kidney Tumor", "Liver Tumor")}}
assert ORGAN[25] == "Kidney Tumor" and ORGAN[26] == "Liver Tumor" and len(ORGAN) == NUM_CLASS
txt = torch.load(f"{repo}/pretrained_weights/txt_encoding.pth", map_location="cpu"); R["txt_encoding_pth_shape"] = list(txt.shape)


def load(backbone, ckpt):
    m = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone=backbone, encoding="word_embedding")
    store = m.state_dict(); ck = torch.load(ckpt, map_location="cpu"); ld = ck["net"]; n = 0; remapped = {}
    for k, v in ld.items():                                     # the key remapping of pred_pseudo.py
        name = ("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])
        remapped[name] = v; store[name] = v; n += 1
    miss, unexp = m.load_state_dict(store, strict=False)
    shape_bad = [(k, tuple(v.shape), tuple(m.state_dict()[k].shape)) for k, v in remapped.items() if k in m.state_dict() and tuple(v.shape) != tuple(m.state_dict()[k].shape)]
    info = {"checkpoint_keys": len(ld), "params_loaded": n, "model_state_keys": len(m.state_dict()), "missing_keys": list(miss)[:10], "unexpected_keys": list(unexp)[:10], "shape_mismatches": shape_bad[:10], "n_parameters_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
            "checkpoint_epoch": ck.get("epoch", None), "top_level_keys": list(ck.keys())}
    emb = remapped.get("organ_embedding"); info["organ_embedding_in_checkpoint"] = emb is not None
    if emb is not None:
        info["organ_embedding_shape"] = list(emb.shape); info["organ_embedding_equals_txt_encoding_pth"] = bool(torch.allclose(emb.float(), txt.float(), atol=1e-6))
    return m.eval(), info


m, R["swin_unetr"] = load("swinunetr", f"{wdir}/swin_unetr.pth")
cap = {}
m.backbone.swinViT.register_forward_hook(lambda mod, a, o: cap.__setitem__("hs", [tuple(t.shape) for t in o]))
with torch.no_grad():
    y = m(torch.zeros(1, 1, 96, 96, 96))
R["swin_unetr"]["output_shape_dummy_96"] = list(y.shape); R["swin_unetr"]["swinViT_hidden_states_shapes_dummy_96"] = cap["hs"]
R["swin_unetr"]["hook_candidate"] = "backbone.swinViT output[2]: %s -> 192 channels, stride %d" % (cap["hs"][2], 96 // cap["hs"][2][-1])
mu, R["unet"] = load("unet", f"{wdir}/unet.pth")
with torch.no_grad():
    yu = mu(torch.zeros(1, 1, 96, 96, 96))
R["unet"]["output_shape_dummy_96"] = list(yu.shape)
json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1))
