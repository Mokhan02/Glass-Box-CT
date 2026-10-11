"""Run the VISTA3D automatic head on one CT and return per-class probabilities in the original image space.

Mirrors the HF pipeline (vista3d_pipeline.py): same preprocessing, SlidingWindowInfererAdapt over 128^3
patches with 0.3 overlap, one class prompt per output channel, Invertd back to the input geometry.
Weights: nvidia/NV-Segment-CT `vista3d_pretrained_model/model.pt` (network: monai vista3d132).
"""
from __future__ import annotations

from collections.abc import Sequence

from glassbox.data.preprocess import PATCH_SIZE, SW_OVERLAP, build_transforms

NETWORK = "vista3d132"  # the bundle network; the paper describes the 127-class release


def load_network(weights_path: str, device):
    import torch
    from monai.networks.nets import vista3d132

    net = vista3d132(encoder_embed_dim=48, in_channels=1)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return net.to(device).eval()


def predict_probs(image_path: str, class_ids: Sequence[int], network, device, image_key: str = "image"):
    """Return (probs, meta): probs is [len(class_ids), *orig_shape] sigmoid probabilities."""
    import torch
    from monai.inferers import SlidingWindowInfererAdapt
    from monai.transforms import Invertd

    pre = build_transforms(image_key)
    data = pre({image_key: image_path})
    img = data[image_key].unsqueeze(0).to(device)  # [1,1,H,W,D]
    prompts = torch.tensor(list(class_ids), device=device).unsqueeze(-1)  # [N,1]
    sw = SlidingWindowInfererAdapt(
        roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate"
    )
    with torch.no_grad():
        logits = sw(
            img, network, transpose=True, point_coords=None, point_labels=None,
            class_vector=prompts, prompt_class=None, prev_mask=None, labels=None, label_set=None,
        )  # [N,H,W,D] or [1,N,H,W,D]
    logits = logits.reshape(len(class_ids), *img.shape[2:]).float().cpu()
    probs = torch.sigmoid(logits)
    # Put the probabilities back on the preprocessed image's metadata and invert to the original geometry.
    from monai.data import MetaTensor

    pred = MetaTensor(probs, meta=data[image_key].meta, applied_operations=data[image_key].applied_operations)
    out = Invertd(
        keys="pred", transform=pre, orig_keys=image_key, nearest_interp=False, to_tensor=True
    )({"pred": pred, image_key: data[image_key]})
    return out["pred"].as_tensor(), out["pred"].meta


def dice(a, b, eps: float = 1e-8) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + eps))


def predict_with_tokens(image_path: str, class_ids: Sequence[int], network, device, image_key: str = "image"):
    """One forward pass that returns everything RQ2 needs for a case:
      probs_orig   [N, *orig_shape]  sigmoid probabilities in the input image's geometry (for Dice)
      probs_native [N, *prep_shape]  the same in the preprocessed space (1.5 mm, RAS, cropped), aligned with token coordinates
      tokens       dict from ActivationRecorder.collect() (mid / bottleneck / decoder_F token tables, no ground-truth labels)
    Air tokens are subsampled to 10% by the recorder; they lie outside any predicted organ, so pooling inside a mask is unaffected."""
    import torch
    from monai.data import MetaTensor
    from monai.inferers import SlidingWindowInfererAdapt
    from monai.transforms import Invertd

    from glassbox.extract.hooks import ActivationRecorder

    pre = build_transforms(image_key)
    data = pre({image_key: image_path})
    img = data[image_key].unsqueeze(0).to(device)
    prompts = torch.tensor(list(class_ids), device=device).unsqueeze(-1)
    rec = ActivationRecorder(network, torch.zeros(img.shape[2:], dtype=torch.long), n_labels=1, seed=0)
    rec.label = rec.label.to(device)
    sw = SlidingWindowInfererAdapt(
        roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate"
    )
    try:
        with torch.no_grad():
            logits = sw(
                img, network, transpose=True, point_coords=None, point_labels=None,
                class_vector=prompts, prompt_class=None, prev_mask=None, labels=None, label_set=None,
            )
        tokens = rec.collect()
    finally:
        rec.close()
    native = torch.sigmoid(logits.reshape(len(class_ids), *img.shape[2:]).float().cpu())
    pred = MetaTensor(native.clone(), meta=data[image_key].meta, applied_operations=data[image_key].applied_operations)
    out = Invertd(keys="pred", transform=pre, orig_keys=image_key, nearest_interp=False, to_tensor=True)(
        {"pred": pred, image_key: data[image_key]}
    )
    return out["pred"].as_tensor(), native, tokens
