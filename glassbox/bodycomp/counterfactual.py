"""Intensity-only lesion edit for the contrast counterfactual (results/contrast_cf/PREREG.md).
A constant HU offset d = target_mean - T is added INSIDE the lesion mask only, with weight 1 on interior voxels and 0.5 on the 1-voxel inner border (6-connected erosion);
edited values are clipped to [-1024, 3071] HU. Nothing outside the mask changes."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

HU_MIN, HU_MAX = -1024.0, 3071.0


def feather_weights(mask: np.ndarray) -> np.ndarray:
    interior = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return np.where(interior, 1.0, np.where(mask, 0.5, 0.0)).astype(np.float32)


def edit_lesion(ct: np.ndarray, mask: np.ndarray, target_mean: float) -> tuple[np.ndarray, dict]:
    """Return (edited copy of ct, info). `ct` is not modified."""
    T = float(ct[mask].mean())
    d = target_mean - T
    w = feather_weights(mask)
    out = ct.astype(np.float32, copy=True)
    out[mask] = np.clip(ct[mask].astype(np.float32) + d * w[mask], HU_MIN, HU_MAX)
    realised = float(out[mask].mean())
    return out, {"T": T, "offset": d, "target_mean": float(target_mean), "realised_mean": realised, "flagged": abs(realised - target_mean) > 5.0}


def arm_targets(T: float, L: float):
    """PREREG targets for a lesion with mean T and local liver mean L (c = T - L): (arm, edit target, control target) or None."""
    c = T - L
    if c < 0:
        return "A", L + abs(c), T - 2 * abs(c)  # dark -> bright; control: darker by the same absolute shift
    if c > 0:
        return "B", L - abs(c), T + 2 * abs(c)  # bright -> dark; control: brighter by the same absolute shift
    return None
