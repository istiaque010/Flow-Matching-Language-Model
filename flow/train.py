"""Train the flow-matching velocity Transformer (Phases 6 and 7).

    python -m flow.train --run main                       # settings from configs/flow.yaml
    python -m flow.train --run abl_notime --steps 2000 --time_cond none
    python -m flow.train --run tiny --overfit 1 --steps 300  # debugging: overfit one batch

Every step: draw B sequences, x1 = frozen standardised embeddings (no gradient into the table), x0 ~ N(0, I),
t ~ U(0, 1) per sequence. With probability p_prompt the sequence gets a clean prefix of k ~ U{L/8 .. 3L/8} tokens
(the prompt convention of Phase 11: prefix positions are set to x1 in x_t, marked by the type embedding, and carry no
loss). Loss = masked MSE between v(x_t, t) and x1 - x0 over valid, non-prefix positions.
Optional auxiliary token loss (ablation, --aux_ce lambda): the implied clean sample x1_hat = x_t + (1 - t) v is scored
against every token with logits = -||x1_hat - e_k||^2 / 2 and a cross-entropy to the true token is added, weighted by
lambda. This changes the objective; it is not equivalent to flow matching.

AdamW, linear warm-up then cosine decay to 0.1 x lr, gradient clipping. Validation (fixed val sequences, fixed noise,
fixed grid of 8 t values): unconditional MSE, prompted MSE (16-token prefix), and the accuracy of decoding x1_hat to
the true token at t = 0.05 / 0.1 / 0.2 / 0.3 (later t decode ~perfectly even untrained). The checkpoint with the lowest unconditional validation MSE is kept.
Logs results/metrics/train/<run>/log.csv and summary.json; checkpoints/<run>/best.pt and last.pt.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from . import CHECKPOINTS, DATA, METRICS, load_config, save_json
from .dataset import valid_mask
from .embeddings import TokenSpace
from .flow_path import build_xt, implied_x1, masked_mse, sample_t, target_velocity
from .model import FlowConfig, VelocityTransformer, count_parameters

VAL_T = torch.tensor([0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95])
ACC_T = (0.05, 0.1, 0.2, 0.3)     # at t >= 0.5 decoding x1_hat is ~100% even for an untrained model (tokens are far apart)


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * p)))


def prefix_masks(B, L, p_prompt, g, fixed_k=None):
    m = torch.zeros(B, L, dtype=torch.bool)
    lo, hi = L // 8, (3 * L) // 8
    for b in range(B):
        if fixed_k is not None:
            m[b, :fixed_k] = True
        elif torch.rand(1, generator=g).item() < p_prompt:
            k = int(torch.randint(lo, hi + 1, (1,), generator=g))
            m[b, :k] = True
    return m


def aux_ce(ts, x1_hat, ids, mask):
    logits = -0.5 * ((x1_hat ** 2).sum(-1, keepdim=True) - 2 * x1_hat @ ts.table.T + ts.table_sq)
    return F.cross_entropy(logits[mask], ids[mask])


def loss_fn(model, ts, ids, g, p_prompt, aux_lambda, L, t_power=1.0):
    valid = torch.from_numpy(valid_mask(ids.numpy()))
    x1 = ts.encode(ids)
    x0 = torch.randn(x1.shape, generator=g)
    t = sample_t(ids.size(0), g, power=t_power)
    pm = prefix_masks(ids.size(0), L, p_prompt, g)
    xt = build_xt(x1, x0, t, pm)
    v = model(xt, t, pm, valid)
    lm = valid & ~pm
    mse = masked_mse(v, target_velocity(x0, x1), lm)
    if aux_lambda > 0:
        ce = aux_ce(ts, implied_x1(xt, v, t), ids, lm)
        return mse + aux_lambda * ce, mse, ce
    return mse, mse, None


@torch.no_grad()
def validate(model, ts, val_ids, L, batch=250):
    model.eval()
    g = torch.Generator().manual_seed(1234)
    tot = {"mse_uncond": 0.0, "mse_prompt16": 0.0}
    acc = {t: 0.0 for t in ACC_T}
    n = 0
    for i in range(0, len(val_ids), batch):
        ids = val_ids[i:i + batch, :L]
        B = ids.size(0)
        valid = torch.from_numpy(valid_mask(ids.numpy()))
        x1 = ts.encode(ids)
        x0 = torch.randn(x1.shape, generator=g)
        t = VAL_T.repeat(B // len(VAL_T) + 1)[:B]
        for key, k in (("mse_uncond", None), ("mse_prompt16", L // 4)):
            pm = prefix_masks(B, L, 0, g, fixed_k=k) if k else torch.zeros(B, L, dtype=torch.bool)
            xt = build_xt(x1, x0, t, pm)
            v = model(xt, t, pm, valid)
            tot[key] += float(masked_mse(v, target_velocity(x0, x1), valid & ~pm)) * B
        for tv in ACC_T:
            tt = torch.full((B,), tv)
            xt = build_xt(x1, x0, tt, None)
            x1h = implied_x1(xt, model(xt, tt, None, valid), tt)
            acc[tv] += float((ts.nearest(x1h)[valid] == ids[valid]).float().mean()) * B
        n += B
    model.train()
    out = {k: v / n for k, v in tot.items()}
    out.update({f"acc_t{t}": a / n for t, a in acc.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="flow.yaml")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--space", choices=["p1small", "pca128"])
    ap.add_argument("--seq_len", type=int)
    ap.add_argument("--time_cond", choices=["adaln", "none"])
    ap.add_argument("--aux_ce", type=float)
    ap.add_argument("--t_power", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--evals", type=int)
    ap.add_argument("--overfit", type=int, default=0, help="train on this many fixed batches (debug)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tc, mc = cfg["train"], cfg["model"]
    for k in ("steps", "lr", "space", "seq_len", "time_cond", "aux_ce", "evals", "t_power", "batch_size"):
        v = getattr(a, k)
        if v is not None:
            (mc if k in ("seq_len", "time_cond") else tc)[k] = v
    torch.manual_seed(a.seed)
    torch.set_num_threads(tc.get("threads", os.cpu_count()))
    g = torch.Generator().manual_seed(1000 + a.seed)
    ts = TokenSpace(tc["space"])
    L = mc["seq_len"]
    mcfg = FlowConfig(d_data=ts.dim, seq_len=L, n_layer=mc["n_layer"], n_head=mc["n_head"], width=mc["width"],
                      dropout=mc["dropout"], time_cond=mc["time_cond"])
    model = VelocityTransformer(mcfg)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    nodecay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}], lr=tc["lr"], betas=tuple(tc["betas"]))
    train = np.load(os.path.join(DATA, "train.npy"), mmap_mode="r")
    val_ids = torch.from_numpy(np.load(os.path.join(DATA, "val.npy")).astype(np.int64))
    rng = np.random.default_rng(a.seed)
    fixed = [torch.from_numpy(train[np.sort(rng.choice(len(train), tc["batch_size"], replace=False)), :L].astype(np.int64))
             for _ in range(a.overfit)]
    run_dir = os.path.join(CHECKPOINTS, a.run)
    log_dir = os.path.join(METRICS, "train", a.run)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    eval_every = max(1, tc["steps"] // tc["evals"])
    proc = psutil.Process()
    peak = 0.0
    best, best_step, rows = float("inf"), -1, []
    t_start = time.perf_counter()
    print(f"[{a.run}] params {count_parameters(model)['total']:,}  space {ts.kind} d={ts.dim}  L={L}  "
          f"steps {tc['steps']}  batch {tc['batch_size']}  lr {tc['lr']}  time {mcfg.time_cond}  aux_ce {tc['aux_ce']}  t_power {tc.get('t_power', 1.0)}")
    model.train()
    for step in range(tc["steps"]):
        lr = lr_at(step, tc)
        for gr in opt.param_groups:
            gr["lr"] = lr
        if fixed:
            ids = fixed[step % len(fixed)]
        else:
            ids = torch.from_numpy(train[np.sort(rng.integers(0, len(train), tc["batch_size"])), :L].astype(np.int64))
        loss, mse, ce = loss_fn(model, ts, ids, g, tc["p_prompt"], tc["aux_ce"], L, tc.get("t_power", 1.0))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]))
        opt.step()
        if step % 25 == 0:
            peak = max(peak, proc.memory_info().rss / 2 ** 20)
        row = {"step": step + 1, "lr": lr, "loss": loss.item(), "mse": mse.item(), "ce": ce.item() if ce is not None else "",
               "grad_norm": gn, "secs": time.perf_counter() - t_start}
        if (step + 1) % eval_every == 0 or step + 1 == tc["steps"]:
            v = validate(model, ts, val_ids, L)
            row.update(v)
            if v["mse_uncond"] < best:
                best, best_step = v["mse_uncond"], step + 1
                torch.save({"model": model.state_dict(), "config": mcfg.to_dict(), "train": tc, "space": ts.kind,
                            "step": step + 1, "val": v}, os.path.join(run_dir, "best.pt"))
            tok_s = (step + 1) * tc["batch_size"] * L / (time.perf_counter() - t_start)
            print(f"[{a.run}] step {step + 1}/{tc['steps']} loss {loss.item():.4f} val mse {v['mse_uncond']:.4f} "
                  f"prompt {v['mse_prompt16']:.4f} acc@.1 {v['acc_t0.1']:.3f} acc@.3 {v['acc_t0.3']:.3f} "
                  f"{tok_s:.0f} tok/s", flush=True)
        rows.append(row)
    secs = time.perf_counter() - t_start
    torch.save({"model": model.state_dict(), "config": mcfg.to_dict(), "train": tc, "space": ts.kind,
                "step": tc["steps"], "val": rows[-1]}, os.path.join(run_dir, "last.pt"))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1].keys()).index(k) if k in rows[-1] else 99)
    with open(os.path.join(log_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    save_json({"run": a.run, "seed": a.seed, "model": mcfg.to_dict(), "train": tc, "params": count_parameters(model),
               "train_seconds": secs, "tokens": tc["steps"] * tc["batch_size"] * L,
               "tokens_per_s": tc["steps"] * tc["batch_size"] * L / secs, "peak_rss_mb": peak,
               "best_step": best_step, "best_val_mse": best, "final_val": {k: v for k, v in rows[-1].items()
                                                                            if k.startswith(("mse_", "acc_"))},
               "overfit_batches": a.overfit}, log_dir, "summary.json")


if __name__ == "__main__":
    main()
