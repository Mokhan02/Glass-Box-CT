"""Lesion measurement from a binary tumour mask (research measurements only): 3D components, longest axial diameter (largest in-plane Feret distance over the lesion's
axial slices, mm), volume, and a RECIST 1.1-style target suggestion (>= 10 mm; at most 2 per organ). Arrays must be in the canonical orientation (axis 2 = axial)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull


def feret_2d(mask2d: np.ndarray, sx: float, sy: float) -> float:
    pts = np.argwhere(mask2d).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:  # collinear points
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def measure_lesions(mask: np.ndarray, prob: np.ndarray | None, zooms, min_diameter_mm: float = 10.0, max_targets: int = 2) -> list[dict]:
    sx, sy, sz = (float(z) for z in zooms[:3]); lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3))); out = []
    for i in range(1, n + 1):
        comp = lab == i; zs = np.flatnonzero(comp.any(axis=(0, 1)))
        diam = max(feret_2d(comp[:, :, z], sx, sy) for z in zs); zbest = int(max(zs, key=lambda z: comp[:, :, z].sum()))
        out.append({"id": i, "diameter_mm": diam, "volume_cm3": float(comp.sum() * sx * sy * sz / 1000.0), "mean_probability": float(prob[comp].mean()) if prob is not None else None,
                    "best_slice": zbest, "centroid_voxel": [float(c) for c in np.argwhere(comp).mean(0)]})
    out.sort(key=lambda l: -l["diameter_mm"]); k = 0
    for l in out:
        if l["diameter_mm"] >= min_diameter_mm and k < max_targets:
            l["recist"] = "target candidate"; k += 1
        else:
            l["recist"] = "non-target (below 10 mm)" if l["diameter_mm"] < min_diameter_mm else "non-target (more than 2 larger lesions in this organ)"
    return out
