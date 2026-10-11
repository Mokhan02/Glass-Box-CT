"""Print the VISTA3D module tree and the activation shapes at every candidate hook point.
Usage (repo root): python scripts/inspect_model.py [device]   -> also writes cache/module_report.txt"""
import sys

import torch

sys.path.insert(0, ".")
from glassbox.seg.infer import load_network  # noqa: E402

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
net = load_network("cache/weights/model.pt", dev)
lines = [f"torch {torch.__version__}  device {dev}", f"params {sum(p.numel() for p in net.parameters())/1e6:.1f}M"]


def tree(m, name="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    for n, c in m.named_children():
        lines.append("  " * depth + f"{name + '.' if name else ''}{n}: {type(c).__name__}")
        tree(c, f"{name + '.' if name else ''}{n}", depth + 1, max_depth)


tree(net)
lines.append(f"class_embeddings: {tuple(net.class_head.class_embeddings.weight.shape)}")
shapes = {}


def shape_of(o):
    return [shape_of(t) for t in o] if isinstance(o, (list, tuple)) else tuple(o.shape)


def hook(name):
    def f(m, i, o):
        shapes[name] = shape_of(o)
    return f


net.image_encoder.encoder.register_forward_hook(hook("image_encoder.encoder (5 stages)"))
net.image_encoder.register_forward_hook(hook("image_encoder (decoder outputs)"))
net.class_head.image_post_mapping.register_forward_hook(hook("class_head.image_post_mapping (= F)"))
x = torch.randn(1, 1, 128, 128, 128, device=dev)
with torch.no_grad():
    net(x, class_vector=torch.tensor([[1]], device=dev), point_coords=None, point_labels=None)
for k, v in shapes.items():
    lines.append(f"{k}: {v}")
open("cache/module_report.txt", "w").write("\n".join(lines))
print("\n".join(lines))
