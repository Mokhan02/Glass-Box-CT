"""Phase estimates for the clean 89-case cohort from the model's own aorta / portal-vein masks (results/trust/PREREG.md section A). GPU, instance-side.
Writes <out>/phase_estimates.csv: case, route, estimate, aorta/portal-vein HU and voxels, predicted-tumour contrast. Usage: python scripts/phase_cohort.py [--out results/trust]"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, ".")
from glassbox.pipeline.scan import canonical, tumour_liver_contrast  # noqa: E402
from glassbox.seg.infer import load_network, predict_probs  # noqa: E402
from glassbox.trust.phase import estimate_phase  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/trust"); ap.add_argument("--rr", default="cache/rr"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
cases = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz"))[: a.limit or None]
dev = torch.device("cuda"); net = load_network("cache/weights/model.pt", dev); rows = []
for i, c in enumerate(cases, 1):
    route = str(np.load(os.path.join(a.rr, c + ".npz"))["route"]); path = os.path.join(ROOTS[route], c, "ct.nii.gz"); img = nib.load(path)
    probs, _ = predict_probs(path, [1, 6, 17, 26], net, dev); P = probs.numpy()
    cc, zc, _ = canonical(np.asarray(img.dataobj).astype(np.float32), img.affine)
    stack = P[:3]; lab = stack.argmax(0) + 1; lab[stack.max(0) <= 0.5] = 0  # liver / aorta / portal vein, overlaps resolved by the larger probability as in the pipeline
    m = lambda k: canonical((lab == k).astype(np.uint8), img.affine)[0].astype(bool)  # noqa: E731
    ph = estimate_phase(cc, m(2), m(3), m(1), zc); tum = canonical((P[3] > 0.5).astype(np.uint8), img.affine)[0].astype(bool)
    con = tumour_liver_contrast(cc, tum, m(1), zc) if tum.any() else None
    rows.append({"case": c, "route": route, "estimate": ph["estimate"], "aorta_hu": ph["aorta_hu"]["value"] if ph["aorta_hu"] else "", "portal_vein_hu": ph["portal_vein_hu"]["value"] if ph["portal_vein_hu"] else "",
                 "aorta_voxels": ph["aorta_voxels"], "portal_vein_voxels": ph["portal_vein_voxels"], "pred_contrast_hu": "" if con is None else round(con, 1), "pred_tumour_voxels": int(tum.sum())})
    print(f"[{i}/{len(cases)}] {c} {ph['estimate']} aorta {rows[-1]['aorta_hu']} pv {rows[-1]['portal_vein_hu']} contrast {rows[-1]['pred_contrast_hu']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "phase_estimates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
