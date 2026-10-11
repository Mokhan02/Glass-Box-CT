"""BatchTopK sparse autoencoder (Bussmann et al. 2024), as used for the Goodfire-style analysis.

  z     = batch_topk(relu((x - b_dec) @ W_enc + b_enc), k * batch_size)   # top k*B activations over the whole batch
  x_hat = z @ W_dec + b_dec                                               # loss = MSE(x, x_hat)

At inference a per-feature-agnostic scalar `threshold` replaces the batch top-k, so one token's code does not
depend on the rest of the batch. The threshold is the running mean of the smallest kept activation per batch.
Decoder rows are kept at unit norm. Inputs are scaled once so the mean token norm is 1 (`scale`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BatchTopKSAE(nn.Module):
    def __init__(self, d_in: int, expansion: int = 16, k: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n = d_in * expansion
        self.d_in, self.n, self.k = d_in, n, k
        w = torch.randn(n, d_in, generator=g)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w)
        self.W_enc = nn.Parameter(w.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("threshold", torch.zeros(()))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("steps_since_active", torch.zeros(n))

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def encode(self, x: torch.Tensor, use_threshold: bool = False) -> torch.Tensor:
        a = self.pre(x)
        if use_threshold:
            return a * (a > self.threshold)
        kb = min(self.k * x.shape[0], a.numel())
        top = torch.topk(a.flatten(), kb)
        z = torch.zeros_like(a).flatten()
        z[top.indices] = top.values
        return z.view_as(a)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        a = self.pre(x)
        kb = min(self.k * x.shape[0], a.numel())
        top = torch.topk(a.flatten(), kb)
        z = torch.zeros_like(a).flatten()
        z[top.indices] = top.values
        z = z.view_as(a)
        x_hat = self.decode(z)
        with torch.no_grad():
            self.threshold.lerp_(top.values.min(), 0.01 if self.threshold > 0 else 1.0)
            active = (z > 0).any(0)
            self.steps_since_active += 1
            self.steps_since_active[active] = 0
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Drop the gradient component along each unit decoder row so Adam does not fight the renormalisation."""
        if self.W_dec.grad is not None:
            g = self.W_dec.grad
            g -= (g * self.W_dec).sum(1, keepdim=True) * self.W_dec


def fit_scale(sample: torch.Tensor) -> float:
    """Factor that makes the mean token norm 1."""
    return float(1.0 / sample.float().norm(dim=1).mean())


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained, relative to the mean token. 1 - FVU is 'variance explained'."""
    x, x_hat = x.detach(), x_hat.detach()
    return float(((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


def dead_fraction(sae: BatchTopKSAE, window: int = 1000) -> float:
    """Share of features that have not fired in the last `window` training batches."""
    return float((sae.steps_since_active > window).float().mean())


def train(sae: BatchTopKSAE, batches, lr: float = 1e-4, log_every: int = 200, log=print):
    """`batches` yields float tensors [B, d_in] already multiplied by sae.scale."""
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    dev = sae.W_dec.device
    for step, x in enumerate(batches, 1):
        x = x.to(dev)
        x_hat, _ = sae(x)
        loss = ((x - x_hat) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
        if step % log_every == 0:
            log(f"step {step} loss {loss.item():.5f} fvu {fvu(x, x_hat):.4f} dead {dead_fraction(sae):.3f}")
    return sae
