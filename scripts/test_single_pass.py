"""Sanity test for predict_with_tokens on one converted held-out case. Usage: python scripts/test_single_pass.py <case_dir> <cuda|cpu>"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from glassbox.seg.infer import load_network, predict_with_tokens  # noqa: E402

case, dev = sys.argv[1], torch.device(sys.argv[2])
net = load_network("cache/weights/model.pt", dev)
t = time.time()
po, pn, tok = predict_with_tokens(f"{case}/ct.nii.gz", [1, 26], net, dev)
print("time %.0fs" % (time.time() - t), "probs_orig", tuple(po.shape), "native", tuple(pn.shape))
for h in ("bottleneck", "mid", "decoder_F"):
    print(h, tok[h + "_x"].shape, "xyz max", tok[h + "_xyz"].max(0))
xyz = tok["mid_xyz"].astype(int)
ok = (xyz >= 0).all(1) & (xyz < np.array(pn.shape[1:])).all(1)
p = pn[0].numpy()[tuple(xyz[ok].T)]
print("tokens in bounds %.3f | mean liver prob at tokens %.3f | tokens inside predicted liver %.3f" % (ok.mean(), p.mean(), (p > 0.5).mean()))
