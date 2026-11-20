"""Generation for the flow model and the baselines, with one common output format (Phases 8, 9, 11, 13).

Flow model: x(0) ~ N(0, I) of shape (B, L, d); integrate dx/dt = v(x, t) with Euler or Heun; prompt positions are
clamped to the clean prompt embeddings after every step (and marked with the prompt type embedding, the same
convention as in training); decode every continuation vector with the nearest-embedding or projection decoder; cut at
the first <EOS> (<PAD> is removed). Prompted samples keep the prompt exactly (checked).

Baselines (frozen, read-only, through Project 4's code): AR models with the KV-cached sampler
(`bench.generate_ar.ar_generate_ids`, prompt = <BOS> + prompt ids, exactly 48 new tokens unless <EOS>), masked
diffusion models with Project 3's sampler (`bench.generate_diffusion.diffusion_generate_ids`, window <BOS> + prompt +
48 [MASK]). Every system gets the same prompt token ids and the same 48-token output budget.

Record per sample: system, setting, seed, index, prompt_ids, cont_ids (cut at EOS), ended (EOS produced), nfe,
batch latency; flow records also the decoding distances (distance to the nearest token, margin to the second).
"""
from __future__ import annotations

import time

import torch

from . import EOS_ID, PAD_ID
from .embeddings import TokenSpace
from .model import FlowConfig, VelocityTransformer
from .ode_sampler import integrate


def load_flow(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = VelocityTransformer(FlowConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, TokenSpace(ck["space"]), ck


def cut(ids):
    ids = [i for i in ids if i != PAD_ID]
    if EOS_ID in ids:
        return ids[:ids.index(EOS_ID)], True
    return ids, False


@torch.no_grad()
def flow_sample(model, ts, n, steps=32, solver="euler", prompts=None, seed=0, decoder="nearest", L=None):
    """prompts: list of equal-length prefix id lists (or None = unconditional). Returns (records, x_final)."""
    L = L or model.cfg.seq_len
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, L, ts.dim, generator=g)
    clamp, pm, k = None, None, 0
    if prompts is not None:
        k = len(prompts[0])
        assert all(len(p) == k for p in prompts)
        pid = torch.tensor(prompts, dtype=torch.long)
        pm = torch.zeros(n, L, dtype=torch.bool)
        pm[:, :k] = True
        vals = torch.zeros(n, L, ts.dim)
        vals[:, :k] = ts.encode(pid)
        clamp = (pm, vals)

    def vel(x, t):
        return model(x, t, pm, None)

    t0 = time.perf_counter()
    x, nfe, _ = integrate(vel, x0, steps, solver, clamp)
    ids = ts.decode(x, decoder)
    if prompts is not None:
        if decoder == "nearest":   # clamped prompt vectors are exact embeddings -> must decode back exactly
            assert torch.equal(ids[:, :k], pid), "prompt was changed"
        ids[:, :k] = pid           # projection misreads ~0.3% of clean embeddings; the prompt is given, not generated
    dt = time.perf_counter() - t0
    _, d1, d2 = ts.nearest(x, return_dist=True)
    recs = []
    for b in range(n):
        row = ids[b].tolist()
        cont, ended = cut(row[k:])
        recs.append({"prompt_ids": list(prompts[b]) if prompts is not None else [], "cont_ids": cont, "ended": ended,
                     "nfe": nfe, "batch_size": n, "batch_latency_s": dt,
                     "dist_nearest": float(d1[b, k:].mean()), "margin": float((d2[b, k:] - d1[b, k:]).mean())})
    return recs, x


def load_baseline(key):
    from bench.load_models import load_model
    return load_model(key)


AR_SAMPLING = {"topp0.95": dict(method="top_p", top_p=0.95), "temp1.0": dict(method="temperature", temperature=1.0),
               "greedy": dict(method="greedy")}
DIFF_SETTINGS = {"small-S32": dict(steps=32, schedule="linear", select="random", sampling=dict(method="top_k", top_k=50)),
                 "small-S8": dict(steps=8, schedule="linear", select="random", sampling=dict(method="top_k", top_k=50)),
                 "rand-S32": dict(steps=32, schedule="confidence", select="random", sampling=dict(method="top_k", top_k=20))}


@torch.no_grad()
def ar_sample(model, prompts, n_new=48, setting="topp0.95", seed=0):
    from bench.generate_ar import ar_generate_ids
    from . import BOS_ID
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    out, ended, nfe = ar_generate_ids(model, [[BOS_ID] + list(p) for p in prompts], n_new, AR_SAMPLING[setting], g)
    dt = time.perf_counter() - t0
    return [{"prompt_ids": list(p), "cont_ids": o, "ended": e, "nfe": len(o) + int(e), "batch_nfe": nfe,
             "batch_size": len(prompts), "batch_latency_s": dt} for p, o, e in zip(prompts, out, ended)]


@torch.no_grad()
def diffusion_sample(model, prompts, n_new=48, setting="small-S32", seed=0):
    from bench.generate_diffusion import diffusion_generate_ids
    s = DIFF_SETTINGS[setting]
    g = torch.Generator().manual_seed(seed)
    k = len(prompts[0])
    t0 = time.perf_counter()
    x, info = diffusion_generate_ids(model, [list(p) for p in prompts], 1 + k + n_new, s["steps"], s["schedule"],
                                     s["select"], s["sampling"], g)
    dt = time.perf_counter() - t0
    recs = []
    for b, p in enumerate(prompts):
        cont, ended = cut(x[b, 1 + k:].tolist())
        recs.append({"prompt_ids": list(p), "cont_ids": cont, "ended": ended, "nfe": info["nfe"],
                     "batch_size": len(prompts), "batch_latency_s": dt})
    return recs
