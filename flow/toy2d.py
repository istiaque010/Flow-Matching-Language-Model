"""Phase 1 milestone: flow matching on a toy 2D distribution, with the same path, loss and ODE code as the language
model.

    python -m flow.toy2d

Data: a mixture of 8 Gaussians on a circle (radius 4, std 0.3). Noise: N(0, I). A small MLP v(x, t) is trained with
the straight-line flow-matching loss, then samples are drawn with Euler (1, 2, 4, 8, 32, 128 steps) and Heun.
Quality = sliced Wasserstein-1 distance between 4,000 samples and 4,000 fresh data points (100 random projections),
plus the share of samples within 3 std of a mixture centre (mode coverage is shown in the figure).
Writes results/metrics/toy2d.json and results/figures/fig0_toy2d.png.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from . import FIGURES, METRICS, save_json
from .flow_path import interpolate, sample_t, target_velocity
from .ode_sampler import integrate

CENTRES = torch.tensor([[4 * math.cos(2 * math.pi * k / 8), 4 * math.sin(2 * math.pi * k / 8)] for k in range(8)])


def sample_data(n, g):
    c = CENTRES[torch.randint(0, 8, (n,), generator=g)]
    return c + 0.3 * torch.randn(n, 2, generator=g)


class MLP(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 16, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h),
                                 nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        f = torch.arange(8, dtype=x.dtype) + 1
        te = torch.cat([torch.sin(math.pi * f * t[:, None]), torch.cos(math.pi * f * t[:, None])], 1)
        return self.net(torch.cat([x, te], 1))


def sliced_w1(a, b, g, n_proj=100):
    d = torch.randn(n_proj, 2, generator=g)
    d = d / d.norm(dim=1, keepdim=True)
    pa, pb = (a @ d.T).sort(0).values, (b @ d.T).sort(0).values
    return float((pa - pb).abs().mean())


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses, t0 = [], time.perf_counter()
    for step in range(4000):
        x1 = sample_data(512, g)
        x0 = torch.randn(512, 2, generator=g)
        t = sample_t(512, g)
        loss = ((model(interpolate(x0, x1, t), t) - target_velocity(x0, x1)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_s = time.perf_counter() - t0
    model.eval()
    ge = torch.Generator().manual_seed(1)
    ref = sample_data(4000, ge)
    noise = torch.randn(4000, 2, generator=ge)
    res = {"train_steps": 4000, "train_seconds": train_s, "loss_first100": float(np.mean(losses[:100])),
           "loss_last100": float(np.mean(losses[-100:])), "noise_sw1": sliced_w1(noise, ref, torch.Generator().manual_seed(2)),
           "samplers": {}}
    samples = {}
    for solver, steps in [("euler", 1), ("euler", 2), ("euler", 4), ("euler", 8), ("euler", 32), ("euler", 128),
                          ("heun", 4), ("heun", 8)]:
        x, nfe, _ = integrate(lambda x, t: model(x, t), noise, steps, solver)
        near = (torch.cdist(x, CENTRES).min(1).values < 0.9).float().mean()
        res["samplers"][f"{solver}{steps}"] = {"nfe": nfe, "sliced_w1": sliced_w1(x, ref, torch.Generator().manual_seed(2)),
                                               "within_3sd_of_a_mode": float(near)}
        samples[f"{solver}{steps}"] = x
    # data-noise reference: two independent data samples
    res["data_vs_data_sw1"] = sliced_w1(sample_data(4000, ge), ref, torch.Generator().manual_seed(2))
    save_json(res, METRICS, "toy2d.json")
    for k, v in res["samplers"].items():
        print(f"{k:>8} NFE {v['nfe']:>3}  sliced W1 {v['sliced_w1']:.3f}  near a mode {v['within_3sd_of_a_mode']:.3f}")
    print("noise", round(res["noise_sw1"], 3), "data vs data", round(res["data_vs_data_sw1"], 3))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, path = None, integrate(lambda x, t: model(x, t), noise[:300], 32, "euler", return_path=True)[2]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].scatter(*noise[:1500].T, s=2, color="#7F7F7F")
    axs[0].set_title("t = 0: Gaussian noise", fontsize=10)
    P = torch.stack(path)
    for i in range(300):
        axs[1].plot(P[:, i, 0], P[:, i, 1], lw=0.4, color="#0072B2", alpha=0.5)
    axs[1].set_title("learned paths (Euler, 32 steps)", fontsize=10)
    axs[2].scatter(*samples["euler32"][:1500].T, s=2, color="#0072B2")
    axs[2].set_title(f"t = 1: samples, 32 steps (W1 {res['samplers']['euler32']['sliced_w1']:.2f})", fontsize=10)
    axs[3].scatter(*samples["euler1"][:1500].T, s=2, color="#D55E00")
    axs[3].set_title(f"1 step only (W1 {res['samplers']['euler1']['sliced_w1']:.2f})", fontsize=10)
    fig.tight_layout()
    import os
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "fig0_toy2d.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
