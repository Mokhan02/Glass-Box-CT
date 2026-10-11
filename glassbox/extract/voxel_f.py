"""Window-averaged decoder_F (48 ch, 128^3 grid, output of class_head.image_post_mapping) for a whole scan.
The class-26 logit is linear in F (logit = w . F, no bias), and sliding-window blending is a plain average, so the averaged F
reproduces the network's blended logit exactly: mean_k(w . F_k) = w . mean_k(F_k)."""
from __future__ import annotations

import torch

from glassbox.extract.hooks import window_pads


class VoxelFAccumulator:
    def __init__(self, network, spatial_shape, device, channels: int = 48):
        self.before, after = window_pads(tuple(spatial_shape))
        self.shape = tuple(spatial_shape)
        pshape = [s + b + a for s, b, a in zip(spatial_shape, self.before, after)]
        self.acc = torch.zeros(channels, *pshape, device=device)
        self.cnt = torch.zeros(*pshape, device=device)
        self._sl = None
        self._handles = [
            network.register_forward_pre_hook(self._pre, with_kwargs=True),
            network.class_head.image_post_mapping.register_forward_hook(self._f),
        ]

    def _pre(self, module, args, kwargs):
        slices = args[1][0] if len(args) > 1 else kwargs["patch_coords"][0]
        self._sl = tuple(slices[-3:])
        return None

    def _f(self, module, args, out):
        self.acc[(slice(None),) + self._sl] += out[0].float()
        self.cnt[self._sl] += 1

    def close(self):
        for h in self._handles:
            h.remove()

    def result(self):
        """[48, H, W, D] averaged F in the unpadded image grid."""
        sel = tuple(slice(b, b + s) for b, s in zip(self.before, self.shape))
        return self.acc[(slice(None),) + sel] / self.cnt[sel].clamp(min=1)
