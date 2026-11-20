"""ODE samplers for dx/dt = v(x, t) from t = 0 (noise) to t = 1 (data) (Phase 8).

Euler:  x <- x + h * v(x, t)                                     1 function evaluation (NFE) per step
Heun:   x' = x + h * v(x, t);  x <- x + h/2 * (v(x, t) + v(x', t + h))   2 NFE per step, except that the last
        step's corrector at t = 1 is still evaluated (so NFE = 2 * steps). Heun is second-order accurate.

`clamp` = (mask (B, L) bool, values (B, L, d)): positions where mask is True are reset to `values` after every step
(prompt conditioning: the prefix stays exactly the clean prompt embeddings). The velocity function receives the
current t as a (B,) tensor. Every call counts its function evaluations.
"""
from __future__ import annotations

import torch


class Counter:
    def __init__(self, fn):
        self.fn, self.nfe = fn, 0

    def __call__(self, x, t):
        self.nfe += 1
        return self.fn(x, t)


def _apply_clamp(x, clamp):
    if clamp is None:
        return x
    m, vals = clamp
    return torch.where(m.unsqueeze(-1), vals, x)


@torch.no_grad()
def integrate(velocity, x0: torch.Tensor, steps: int, solver: str = "euler", clamp=None, return_path: bool = False):
    """Returns (x at t = 1, number of function evaluations, optional list of intermediate states)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    f = Counter(velocity)
    B = x0.size(0)
    h = 1.0 / steps
    x = _apply_clamp(x0, clamp)
    path = [x.clone()] if return_path else None
    for k in range(steps):
        t = torch.full((B,), k * h, dtype=x.dtype, device=x.device)
        v = f(x, t)
        if solver == "euler":
            x = x + h * v
        elif solver == "heun":
            xp = _apply_clamp(x + h * v, clamp)
            v2 = f(xp, t + h)
            x = x + 0.5 * h * (v + v2)
        else:
            raise ValueError(f"unknown solver {solver}")
        x = _apply_clamp(x, clamp)
        if return_path:
            path.append(x.clone())
    return x, f.nfe, path
