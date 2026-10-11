"""Preprocessing shared by every stage. Mirrors the VISTA3D bundle's inference.json
(Spacingd 1.5 mm iso, CropForegroundd margin 10, ScaleIntensityRanged, RAS).

Two intensity views are kept on purpose:
  - model input: HU clipped to the bundle window and scaled to [0, 1]
  - raw HU: used for body-composition measures (muscle window -29..150 HU), never the scaled view
"""
from __future__ import annotations

SPACING_MM = (1.5, 1.5, 1.5)
HU_MIN, HU_MAX = -963.8247715525971, 1053.678477684517
CROP_MARGIN = 10
PATCH_SIZE = (128, 128, 128)
SW_OVERLAP = 0.3


def build_transforms(image_key: str = "image"):
    """Model-input transforms. MONAI is imported lazily so the rest of the package stays light."""
    from monai import transforms as T

    return T.Compose(
        [
            T.LoadImaged(keys=image_key, image_only=True),
            T.EnsureChannelFirstd(keys=image_key),
            T.Spacingd(keys=image_key, pixdim=SPACING_MM, mode="bilinear"),
            T.CropForegroundd(keys=image_key, source_key=image_key, margin=CROP_MARGIN, allow_smaller=True),
            T.ScaleIntensityRanged(keys=image_key, a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
            T.Orientationd(keys=image_key, axcodes="RAS"),
            T.CastToTyped(keys=image_key, dtype="float32"),
        ]
    )


def build_raw_hu_transforms(image_key: str = "image"):
    """Same geometry as the model input but intensities stay in raw HU."""
    from monai import transforms as T

    return T.Compose(
        [
            T.LoadImaged(keys=image_key, image_only=True),
            T.EnsureChannelFirstd(keys=image_key),
            T.Spacingd(keys=image_key, pixdim=SPACING_MM, mode="bilinear"),
            T.CropForegroundd(keys=image_key, source_key=image_key, margin=CROP_MARGIN, allow_smaller=True),
            T.Orientationd(keys=image_key, axcodes="RAS"),
            T.CastToTyped(keys=image_key, dtype="float32"),
        ]
    )


def build_pair_transforms(image_key: str = "image", label_key: str = "label"):
    """Image + integer label map through the same geometry (labels use nearest). Inputs are MetaTensors already
    loaded, so the label can be assembled from several mask files first. Foreground crop is driven by the image."""
    from monai import transforms as T

    keys = [image_key, label_key]
    return T.Compose(
        [
            T.EnsureChannelFirstd(keys=keys),
            T.Spacingd(keys=keys, pixdim=SPACING_MM, mode=("bilinear", "nearest")),
            T.CropForegroundd(keys=keys, source_key=image_key, margin=CROP_MARGIN, allow_smaller=True),
            T.ScaleIntensityRanged(keys=image_key, a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
            T.Orientationd(keys=keys, axcodes="RAS"),
            T.CastToTyped(keys=[image_key], dtype="float32"),
        ]
    )
