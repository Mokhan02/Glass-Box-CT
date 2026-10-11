"""L3 muscle measures from a CT in raw HU and label masks. Research measurements only.

All arrays are put in nibabel's closest-canonical (RAS+) orientation first, so axis 2 is the axial (S-I) axis.
Muscle voxels are psoas + paraspinal (autochthon) labels on the L3 slice, restricted to -29..150 HU.
"""
from __future__ import annotations

import nibabel as nib
import numpy as np

MUSCLE_HU = (-29.0, 150.0)
LOW_ATTENUATION_HU = 30.0  # voxels below this inside muscle = fat-infiltrated ("low-attenuation muscle")
PSOAS = ("iliopsoas_left", "iliopsoas_right")
PARASPINAL = ("autochthon_left", "autochthon_right")


def load_canonical(path: str):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), tuple(float(z) for z in img.header.get_zooms()[:3])


def l3_slice_index(l3_mask: np.ndarray) -> int | None:
    """Axial slice through the centroid of the L3 vertebra mask (axis 2), or None if L3 is absent."""
    z = np.flatnonzero(l3_mask.any(axis=(0, 1)))
    if len(z) == 0:
        return None
    return int(round(np.average(np.arange(l3_mask.shape[2]), weights=l3_mask.sum(axis=(0, 1)))))


def local_hu_stats(hu: np.ndarray) -> np.ndarray:
    """Local baseline features: summary of HU values inside the L3 muscle region (excludes nothing else)."""
    p = np.percentile(hu, [5, 25, 50, 75, 95])
    return np.array([hu.mean(), hu.std(), *p], dtype=np.float32)


def l3_muscle(ct: np.ndarray, masks: dict[str, np.ndarray], zooms: tuple[float, float, float]) -> dict | None:
    k = l3_slice_index(masks["vertebrae_L3"])
    if k is None:
        return None
    sl = lambda name: masks[name][:, :, k] > 0  # noqa: E731
    region = np.zeros(ct.shape[:2], bool)
    for n in PSOAS + PARASPINAL:
        region |= sl(n)
    hu = ct[:, :, k].astype(np.float32)
    keep = region & (hu >= MUSCLE_HU[0]) & (hu <= MUSCLE_HU[1])
    if keep.sum() < 20:
        return None
    px_cm2 = zooms[0] * zooms[1] / 100.0
    psoas = np.zeros_like(region)
    for n in PSOAS:
        psoas |= sl(n)
    v = hu[keep]
    return {
        "l3_slice": k,
        "muscle_area_cm2": float(keep.sum() * px_cm2),
        "psoas_area_cm2": float((psoas & keep).sum() * px_cm2),
        "muscle_mean_hu": float(v.mean()),
        "muscle_sd_hu": float(v.std()),
        "lama_fraction": float((v < LOW_ATTENUATION_HU).mean()),
        "local_hu": local_hu_stats(v),
    }


def global_hu_histogram(ct: np.ndarray, bins: int = 64) -> np.ndarray:
    """Global baseline: normalised 64-bin histogram of all voxels above air (-1000..1500 HU)."""
    v = ct[(ct > -1000) & (ct < 1500)]
    h, _ = np.histogram(v, bins=bins, range=(-1000, 1500))
    return (h / max(h.sum(), 1)).astype(np.float32)
