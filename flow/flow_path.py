"""The flow-matching path and loss (Phases 4 and 6).

Straight-line conditional path with an independent coupling (x0 and x1 drawn independently, no optimal-transport
pairing):
    x_t = (1 - t) * x0 + t * x1,      t ~ U(0, 1)
    u_t = d x_t / dt = x1 - x0        (the target velocity, constant along each line)
The network v(x_t, t) is trained to predict u_t with a mean squared error over valid (non-padding) positions that
are also flow variables (prompt positions are clean and fixed, so they carry no loss).

Useful identities for diagnostics: the implied clean sample is  x1_hat = x_t + (1 - t) * v  and the implied noise is
x0_hat = x_t - t * v.
"""
from __future__ import annotations

import torch


def sample_t(n: int, generator=None, device="cpu", eps: float = 1e-4, power: float = 1.0) -> torch.Tensor:
    """t = u^power with u ~ U(eps, 1 - eps), one value per sequence (eps avoids the exact end points).
    power = 1: uniform (the main setting); power = 2: more weight on early times (ablation H12)."""
    return (eps + (1 - 2 * eps) * torch.rand(n, generator=generator, device=device)) ** power


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t for t of shape (B,), broadcast over all remaining dimensions."""
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - tt) * x0 + tt * x1


def target_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - x0


def implied_x1(xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    tt = t.view(-1, *([1] * (xt.dim() - 1)))
    return xt + (1 - tt) * v


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of (pred - target)^2 over positions where mask is True and over all feature dimensions.
    pred/target: (B, L, d); mask: (B, L) bool."""
    m = mask.unsqueeze(-1).to(pred.dtype)
    se = ((pred - target) ** 2) * m
    return se.sum() / (m.sum() * pred.size(-1)).clamp_min(1.0)


def build_xt(x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, prefix_mask: torch.Tensor | None):
    """x_t with prompt (prefix) positions replaced by their clean embeddings, the same convention the sampler uses
    (it clamps the prefix at every ODE step)."""
    xt = interpolate(x0, x1, t)
    if prefix_mask is not None:
        xt = torch.where(prefix_mask.unsqueeze(-1), x1, xt)
    return xt
