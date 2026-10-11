"""Landing-page thumbnails: the axial slice through the largest predicted lesion, window -150..250 HU, predicted tumour in the accent colour. Reads the viewer copies
under <demo_dir>/<scan>/ (ct.nii.gz, tumour.nii.gz, lesions.json) and writes thumb.png next to them. Usage: python scripts/make_thumbs.py [data/demo]"""
import glob
import json
import os
import sys

import matplotlib
import nibabel as nib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "data/demo"
for d in sorted(glob.glob(os.path.join(root, "demo_*"))):
    ct_f, tu_f = nib.load(os.path.join(d, "ct.nii.gz")), nib.load(os.path.join(d, "tumour.nii.gz"))
    ct_c, tu_c = nib.as_closest_canonical(ct_f), nib.as_closest_canonical(tu_f)
    ct, tu = np.asarray(ct_c.dataobj).astype(np.float32), np.asarray(tu_c.dataobj) > 0
    les = json.load(open(os.path.join(d, "lesions.json")))["lesions"]
    if les:  # voxel in the file's own array -> canonical array, through world coordinates
        v = np.linalg.inv(ct_c.affine) @ ct_f.affine @ np.array([*les[0]["voxel_in_file"], 1.0]); z = int(np.clip(round(v[2]), 0, ct.shape[2] - 1))
    else:
        z = ct.shape[2] // 2
    sl, ts = ct[:, :, z], tu[:, :, z]
    body = ndimage.binary_opening(sl > -300, iterations=2); xs, ys = np.nonzero(body); pad = 8
    x0, x1, y0, y1 = max(xs.min() - pad, 0), min(xs.max() + pad, sl.shape[0]), max(ys.min() - pad, 0), min(ys.max() + pad, sl.shape[1])
    sl, ts = sl[x0:x1, y0:y1], ts[x0:x1, y0:y1]
    img = np.clip((sl - -150) / 400.0, 0, 1)[::-1, :].T; m = ts[::-1, :].T  # radiological display: patient right on the image left, anterior up
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100, facecolor="#07090D"); ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#07090D")
    ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="bilinear", aspect="equal")
    rgba = np.zeros((*m.shape, 4)); rgba[m] = [0.37, 0.90, 0.82, 0.42]; ax.imshow(rgba, origin="lower", interpolation="nearest", aspect="equal")
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=["#5EE6D0"], linewidths=1.2)
    ax.axis("off"); fig.savefig(os.path.join(d, "thumb.png"), facecolor="#07090D"); plt.close(fig); print("thumb", d, "z", z)
