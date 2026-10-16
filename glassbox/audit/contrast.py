"""Lesion selection and signed tumour-minus-liver contrast, identical to scripts/exp8_lesion_covariates.py (results/w1_training_audit/PREREG.md).

Lesion: 26-connected component of the tumour mask, longest axial (Feret) diameter >= MIN_DIAM_MM, the N_LESIONS largest per case.
Contrast c = mean HU of the lesion minus mean HU of non-tumour liver in the 5-20 mm shell around it (true spacing); if the shell has fewer than
MIN_SHELL_VOX voxels, the mean HU of all non-tumour liver voxels. Bright = c > 0."""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

MIN_DIAM_MM, N_LESIONS, MIN_SHELL_VOX, SHELL_MM = 10.0, 2, 200, (5.0, 20.0)
STRUCT26 = np.ones((3, 3, 3))


def feret(m2, sx, sy):
    pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def select_lesions(tumour, spacing, n_lesions=N_LESIONS, min_diam_mm=MIN_DIAM_MM):
    """Return (label_volume, [(diameter_mm, label_id)]) sorted by diameter, largest first. tumour: bool [X, Y, Z]; spacing: (sx, sy, sz) mm."""
    lab, n = ndimage.label(tumour, structure=STRUCT26)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        zs = np.flatnonzero(comp.any(axis=(0, 1)))
        out.append((max(feret(comp[:, :, z], spacing[0], spacing[1]) for z in zs), i))
    out = sorted([o for o in out if o[0] >= min_diam_mm], key=lambda o: -o[0])
    return lab, (out if n_lesions is None else out[:n_lesions])


def lesion_contrast(ct, lesion, liver, spacing):
    """ct: HU [X, Y, Z]; lesion: bool mask of one component; liver: bool mask of liver tissue (tumour voxels are removed here). Returns a dict."""
    sx, sy, sz = spacing
    liver = liver & ~lesion
    xs, ys, zs = np.nonzero(lesion)
    t = float(ct[lesion].mean())
    pad = int(25 / min(sx, sy, sz)); X, Y, Z = ct.shape
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, X)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, Y)
    z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad + 1, Z)
    les3 = lesion[x0:x1, y0:y1, z0:z1]
    dist = ndimage.distance_transform_edt(~les3, sampling=(sx, sy, sz))
    shell = liver[x0:x1, y0:y1, z0:z1] & (dist >= SHELL_MM[0]) & (dist <= SHELL_MM[1])
    n_shell, n_liver = int(shell.sum()), int(liver.sum())
    if n_liver == 0:
        return {"tumour_hu": t, "liver_hu": float("nan"), "contrast_hu": float("nan"), "n_shell_vox": n_shell, "fallback": True, "n_liver_vox": 0}
    fallback = n_shell < MIN_SHELL_VOX
    L = float(ct[liver].mean()) if fallback else float(ct[x0:x1, y0:y1, z0:z1][shell].mean())
    return {"tumour_hu": t, "liver_hu": L, "contrast_hu": t - L, "n_shell_vox": n_shell, "fallback": fallback, "n_liver_vox": n_liver}


def case_lesions(ct, tumour, liver, spacing, n_lesions=N_LESIONS, min_diam_mm=MIN_DIAM_MM, exclude=None):
    """Rows (one per selected lesion, rank 1 = largest) with diameter, volume and contrast.
    Reference parenchyma = liver & ~tumour & ~exclude, where `tumour` is the mask of ALL tumour voxels of the case (every component, not only the selected
    lesions) and `exclude` is any other labelled non-parenchyma (e.g. the labelled portal vein). This mirrors exp8, whose liver mask is the single-valued label 1,
    which already excludes every tumour component (label 2) and the labelled portal vein (label 3) (DEVIATIONS #3)."""
    ref = liver & ~tumour
    if exclude is not None:
        ref = ref & ~exclude
    liver = ref
    lab, sel = select_lesions(tumour, spacing, n_lesions, min_diam_mm)
    vox = spacing[0] * spacing[1] * spacing[2]
    rows = []
    for rank, (d, i) in enumerate(sel, 1):
        comp = lab == i
        rows.append({"rank": rank, "diam_mm": d, "volume_mm3": float(comp.sum() * vox), **lesion_contrast(ct, comp, liver, spacing)})
    return rows
