"""Diagnose the continuous-to-discrete gap (Phase 10).

    python -m flow.diagnose            # main model; -> results/metrics/diagnostics.json

1 decodability over time   accuracy of decoding the implied clean sample x1_hat = x_t + (1 - t) v to the TRUE token,
                           on 500 test openings, for t from 0.02 to 0.98 (nearest and projection decoder), next to
                           the same accuracy for the "no model" guess x1_hat = x_t / t (pure rescaling).
2 decision time            along the chosen ODE path (100 test prompts), the share of positions whose current decoded
                           token (of x1_hat) already equals the final token, per step.
3 prompt influence         same starting noise, (a) with the real prompt, (b) without a prompt, (c) with another
                           prompt: share of identical continuation tokens. If the model used the prompt, (a) vs (b)
                           and (a) vs (c) would agree little.
4 end-point geometry       distance of the final vectors to the nearest token and the margin to the second nearest,
                           for clean embeddings (0 / typical gap), noisy clean embeddings, and generated samples.
5 frequent-token overuse   share of generated tokens that are among the 100 most frequent training tokens, and the
                           number of distinct tokens, for flow, AR, MDLM and real continuations (from the generation
                           files of the test stage).
"""
from __future__ import annotations

import collections
import os

import numpy as np
import torch

from . import CHECKPOINTS, DATA, GENERATIONS, METRICS, load_json, read_jsonl, save_json
from .dataset import valid_mask
from .flow_path import build_xt, implied_x1
from .generate import load_flow

T_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]


@torch.no_grad()
def decodability(model, ts):
    ids = torch.from_numpy(np.load(os.path.join(DATA, "test.npy"))[:500].astype(np.int64))
    valid = torch.from_numpy(valid_mask(ids.numpy()))
    x1 = ts.encode(ids)
    g = torch.Generator().manual_seed(7)
    x0 = torch.randn(x1.shape, generator=g)
    out = []
    for tv in T_GRID:
        t = torch.full((len(ids),), tv)
        xt = build_xt(x1, x0, t, None)
        v = torch.cat([model(xt[i:i + 100], t[i:i + 100], None, valid[i:i + 100]) for i in range(0, len(ids), 100)])
        x1h = implied_x1(xt, v, t)
        mse = float(((v - (x1 - x0)) ** 2)[valid].mean())
        out.append({"t": tv, "mse": mse,
                    "acc_nearest": float((ts.nearest(x1h) == ids)[valid].float().mean()),
                    "acc_projection": float((ts.project(x1h) == ids)[valid].float().mean()),
                    "acc_rescaled_xt": float((ts.nearest(xt / tv) == ids)[valid].float().mean())})
    return out


@torch.no_grad()
def decision_time(model, ts, steps, solver):
    a = np.load(os.path.join(DATA, "test_prompt.npy")).astype(np.int64)[:100]
    P = torch.from_numpy(a[:, :16])
    n, L = len(a), model.cfg.seq_len
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, L, ts.dim, generator=g)
    pm = torch.zeros(n, L, dtype=torch.bool)
    pm[:, :16] = True
    vals = torch.zeros(n, L, ts.dim)
    vals[:, :16] = ts.encode(P)
    x = torch.where(pm.unsqueeze(-1), vals, x)
    h = 1.0 / steps
    traj = []
    for k in range(steps):
        t = torch.full((n,), k * h)
        v = model(x, t, pm, None)
        traj.append(ts.nearest(implied_x1(x, v, t))[:, 16:])
        if solver == "heun":
            xp = torch.where(pm.unsqueeze(-1), vals, x + h * v)
            v = 0.5 * (v + model(xp, t + h, pm, None))
        x = torch.where(pm.unsqueeze(-1), vals, x + h * v)
    final = ts.nearest(x)[:, 16:]
    return [{"t": k * h, "share_equal_final": float((tr == final).float().mean())} for k, tr in enumerate(traj)]


@torch.no_grad()
def prompt_influence(model, ts, steps, solver, decoder):
    from .generate import flow_sample
    a = np.load(os.path.join(DATA, "test_prompt.npy")).astype(np.int64)[:200]
    P = a[:, :16].tolist()
    other = P[1:] + P[:1]
    ra, _ = flow_sample(model, ts, 200, steps, solver, P, 11, decoder)
    rb, _ = flow_sample(model, ts, 200, steps, solver, None, 11, decoder)
    rc, _ = flow_sample(model, ts, 200, steps, solver, other, 11, decoder)
    rd, _ = flow_sample(model, ts, 200, steps, solver, P, 12, decoder)     # same prompt, different noise

    def agree(x, y, off_y=0):
        s = []
        for p, q in zip(x, y):
            u, w = p["cont_ids"][:48], q["cont_ids"][off_y:off_y + 48]
            m = min(len(u), len(w))
            s.append(np.mean([u[i] == w[i] for i in range(m)]) if m else 0.0)
        return float(np.mean(s))
    return {"same_noise_prompt_vs_no_prompt": agree(ra, rb, 16),
            "same_noise_prompt_vs_other_prompt": agree(ra, rc),
            "same_prompt_different_noise": agree(ra, rd),
            "note": "share of identical tokens at the same continuation position (positions 16-63)"}


@torch.no_grad()
def geometry(ts):
    ids = torch.from_numpy(np.load(os.path.join(DATA, "test.npy"))[:200].astype(np.int64)).reshape(-1)
    x = ts.encode(ids)
    _, d1, d2 = ts.nearest(x, return_dist=True)
    g = torch.Generator().manual_seed(5)
    _, n1, n2 = ts.nearest(x + torch.randn(x.shape, generator=g), return_dist=True)
    return {"clean": {"dist_nearest": float(d1.mean()), "margin": float((d2 - d1).mean())},
            "clean_plus_N(0,1)_noise": {"dist_nearest": float(n1.mean()), "margin": float((n2 - n1).mean())}}


def overuse():
    from .embeddings import TokenSpace
    top = set(TokenSpace("p1small").freq.argsort(descending=True)[:100].tolist())
    runs = load_json(METRICS, "runs.json")
    ch = load_json(METRICS, "flow_chosen.json")
    files = {"real": "test_real", "flow": f"test_flow_{ch['solver']}{ch['steps']}_{ch['decoder']}",
             "ar5": "test_ar5_topp0.95_s0", "mdlm5": "test_mdlm5_small-S32_s0", "unigram": "test_unigram"}
    out = {}
    for k, f in files.items():
        if f not in runs:
            continue
        rows = read_jsonl(GENERATIONS, f"{f}.jsonl")
        toks = [t for r in rows for t in r["cont_ids"]]
        c = collections.Counter(toks)
        out[k] = {"share_top100_tokens": float(np.mean([t in top for t in toks])), "distinct_tokens": len(c),
                  "most_common": c.most_common(8)}
    return out


def main():
    torch.set_num_threads(os.cpu_count())
    model, ts, _ = load_flow(os.path.join(CHECKPOINTS, "main", "best.pt"))
    ch = load_json(METRICS, "flow_chosen.json")
    out = {"decodability": decodability(model, ts)}
    out["decision_time"] = decision_time(model, ts, ch["steps"], ch["solver"])
    out["prompt_influence"] = prompt_influence(model, ts, ch["steps"], ch["solver"], ch["decoder"])
    out["geometry"] = geometry(ts)
    runs = load_json(METRICS, "runs.json")
    out["geometry"]["generated_by_steps"] = {k: {"dist_nearest": v["dist_nearest"], "margin": v["margin"]}
                                             for k, v in runs.items() if k.startswith("test_flow_") and "dist_nearest" in v
                                             and k.endswith("nearest")}
    out["overuse"] = overuse()
    save_json(out, METRICS, "diagnostics.json")
    for r in out["decodability"]:
        print(f"t {r['t']:.2f}  mse {r['mse']:.3f}  acc nearest {r['acc_nearest']:.3f}  projection {r['acc_projection']:.3f}"
              f"  rescaled x_t {r['acc_rescaled_xt']:.3f}")
    print("decision", [(round(d["t"], 2), round(d["share_equal_final"], 3)) for d in out["decision_time"][::4]])
    print(out["prompt_influence"])
    print(out["geometry"]["clean"], out["geometry"]["clean_plus_N(0,1)_noise"])
    for k, v in out["overuse"].items():
        print(k, round(v["share_top100_tokens"], 3), v["distinct_tokens"])


if __name__ == "__main__":
    main()
