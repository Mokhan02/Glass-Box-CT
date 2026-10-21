"""W3 Stage 2 helpers (pure numpy / scipy, no torch): ground-truth token labels on a model's own preprocessed grid, in the definitions of results/voxel_probe/PREREG.md (VISTA-3D) with anisotropic token cells.
Label map convention (as scripts/voxel_probe_extract.py): 0 background, 1 liver (non-tumour), 2 tumour, 3 unknown (slice without annotation). Arrays are in the model grid's own axis order; `spacing` (mm per voxel) and `cell` (token cell size in voxels) follow the same order."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

INTERIOR_MM = 2.0
ROI_MM = 10.0


def integral(v):
    II = np.zeros([s + 1 for s in v.shape], np.float64); II[1:, 1:, 1:] = v.astype(np.float64).cumsum(0).cumsum(1).cumsum(2); return II


def box_sums(II, lo, hi):
    """Sum over [lo, hi) boxes from a zero-padded integral image; lo / hi [N, 3] already clipped."""
    x0, y0, z0 = lo.T; x1, y1, z1 = hi.T
    return II[x1, y1, z1] - II[x0, y1, z1] - II[x1, y0, z1] - II[x1, y1, z0] + II[x0, y0, z1] + II[x0, y1, z0] + II[x1, y0, z0] - II[x0, y0, z0]


def roi_mask(L, spacing):
    """Voxels within 10 mm of the GT liver + tumour (token centres must lie inside)."""
    return ndimage.distance_transform_edt(~((L == 1) | (L == 2)), sampling=spacing) <= ROI_MM


class GridStats:
    """Integral images of the label map, the predicted tumour mask and the HU volume on one model grid; `edt` = distance (mm) to the nearest KNOWN non-tumour voxel."""

    def __init__(self, L, pred, hu, spacing):
        self.shape = np.array(L.shape); tum, liv, known = L == 2, L == 1, L != 3
        self.II = {k: integral(v) for k, v in (("tum", tum), ("liv", liv), ("known", known), ("pred", pred), ("one", np.ones_like(tum)))}
        self.Ihu1, self.Ihu2 = integral(hu), integral(hu.astype(np.float64) ** 2); self.hu = hu
        self.edt = ndimage.distance_transform_edt(tum | ~known, sampling=spacing)

    def tokens(self, xyz, cell):
        """Per token (centre voxel `xyz` [N, 3], cell size `cell` voxels per axis): GT tumour / liver coverage over KNOWN voxels, predicted-tumour coverage, interior distance, group masks and 14 local HU features."""
        xyz = np.asarray(xyz, np.int64); cell = np.asarray(cell, np.int64); S = self.shape
        lo = np.clip(xyz - cell // 2, 0, S); hi = np.clip(lo + cell, 0, S); n_cell = box_sums(self.II["one"], lo, hi); n_known = box_sums(self.II["known"], lo, hi)
        s_tum, s_liv, s_pred = (box_sums(self.II[k], lo, hi) for k in ("tum", "liv", "pred"))
        full = n_cell >= 0.5 * int(np.prod(cell)); kf = n_known / np.maximum(n_cell, 1); ct = s_tum / np.maximum(n_known, 1); cl = s_liv / np.maximum(n_known, 1); pc = s_pred / np.maximum(n_cell, 1)
        is_t = full & (kf >= 0.5) & (ct >= 0.5); is_l = full & (kf >= 0.5) & (s_tum == 0) & (cl >= 0.5); c = np.clip(xyz, 0, S - 1)
        return {"is_tum": is_t, "is_liv": is_l, "cov_tum": ct, "cov_liv": cl, "pred_cov": pc, "edt": self.edt[c[:, 0], c[:, 1], c[:, 2]], "hu14": self.hu14(xyz, cell)}

    def hu14(self, xyz, cell):
        """Cell and 3x-cell cube: mean, std, p5, p25, p50, p75, p95 of HU (the VISTA PREREG feature set)."""
        S = self.shape; hf = np.zeros((len(xyz), 14), np.float32)
        for scale, off in ((1, 0), (3, 7)):
            size = cell * scale; half = size // 2; lo = np.clip(xyz - half, 0, S); hi = np.clip(xyz + (size - half), 0, S); n = np.maximum(box_sums(self.II["one"], lo, hi), 1)
            m = box_sums(self.Ihu1, lo, hi) / n; v = box_sums(self.Ihu2, lo, hi) / n - m ** 2; hf[:, off] = m; hf[:, off + 1] = np.sqrt(np.maximum(v, 0))
            for k, (a, b) in enumerate(zip(lo, hi)):
                hf[k, off + 2:off + 7] = np.percentile(self.hu[a[0]:b[0], a[1]:b[1], a[2]:b[2]], [5, 25, 50, 75, 95])
        return hf


def window_origins(shape, roi, overlap, scan_interval=None):
    """Start voxels of the sliding windows (the MONAI scan-interval grid: interval = int(roi * (1 - overlap)), last window flush with the end). Used by the token recorders; the prediction itself is made by the models' own inferers."""
    out = []
    for n, r in zip(shape, roi):
        if n <= r:
            out.append([0]); continue
        step = max(int(r * (1 - overlap)), 1) if scan_interval is None else scan_interval; starts = list(range(0, n - r + 1, step))
        if starts[-1] != n - r:
            starts.append(n - r)
        out.append(starts)
    return [(a, b, c) for a in out[0] for b in out[1] for c in out[2]]
