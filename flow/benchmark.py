"""Speed, memory and compute of every generator on this CPU (Phases 12 and 13).

    python -m flow.benchmark          # -> results/metrics/benchmark.json

Task: 16-token prompt, 48 new tokens (AR stops only at the budget here: <EOS> is not used, so every system makes
exactly 48 tokens). All threads (24), fp32, models loaded once; 3 warm-up calls before timing.
  latency      batch 1, 30 test prompts: median and 95th percentile seconds per continuation; tokens/s = 48 / median
  throughput   batch 64: 3 timed batches, new tokens per second
  memory       peak extra process memory (RSS, sampled every 5 ms) during a batch-64 call, above the level after
               loading the model
  NFE          network calls per continuation; FLOPs per continuation ~ 2 x parameters x positions processed
               (flow: 64 positions x NFE; MDLM: 65 x steps; AR with KV cache: 16 + 48 positions, once each;
               attention terms are left out, so this is a lower bound)
"""
from __future__ import annotations

import os
import statistics
import threading
import time

import numpy as np
import psutil
import torch

from . import CHECKPOINTS, DATA, METRICS, load_json, save_json
from .generate import load_baseline, load_flow

N_NEW = 48


class PeakRSS:
    def __init__(self):
        self.p = psutil.Process()
        self.peak = 0
        self.run = False

    def __enter__(self):
        self.base = self.p.memory_info().rss
        self.peak = self.base
        self.run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def _loop(self):
        while self.run:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.005)

    def __exit__(self, *a):
        self.run = False
        self.t.join()

    @property
    def extra_mb(self):
        return (self.peak - self.base) / 2 ** 20


def gen_fn(kind, model, ts=None, steps=None, solver=None, setting=None):
    from .generate import AR_SAMPLING, DIFF_SETTINGS, flow_sample
    from . import BOS_ID
    if kind == "flow":
        return lambda P, seed: flow_sample(model, ts, len(P), steps, solver, P, seed, "nearest")
    if kind == "ar":
        from bench.generate_ar import ar_generate_ids

        def f(P, seed):
            g = torch.Generator().manual_seed(seed)
            return ar_generate_ids(model, [[BOS_ID] + p for p in P], N_NEW, AR_SAMPLING[setting], g, stop_at_eos=False)
        return f
    from bench.generate_diffusion import diffusion_generate_ids
    s = DIFF_SETTINGS[setting]

    def f(P, seed):
        g = torch.Generator().manual_seed(seed)
        return diffusion_generate_ids(model, P, 1 + len(P[0]) + N_NEW, s["steps"], s["schedule"], s["select"],
                                      s["sampling"], g)
    return f


def bench(name, fn, prompts, nfe, params, positions):
    for i in range(3):
        fn(prompts[i:i + 1], i)
    lat = []
    for i in range(30):
        t0 = time.perf_counter()
        fn(prompts[i:i + 1], 100 + i)
        lat.append(time.perf_counter() - t0)
    thr = []
    with PeakRSS() as mem:
        for r in range(3):
            t0 = time.perf_counter()
            fn(prompts[64 * r % 300:64 * r % 300 + 64], 500 + r)
            thr.append(64 * N_NEW / (time.perf_counter() - t0))
    med = statistics.median(lat)
    res = {"latency_median_s": med, "latency_p95_s": float(np.percentile(lat, 95)), "tokens_per_s_batch1": N_NEW / med,
           "throughput_batch64_tok_s": statistics.median(thr), "peak_extra_mb_batch64": mem.extra_mb, "nfe": nfe,
           "params": params, "gflops_per_continuation": 2 * params * positions / 1e9}
    print(f"{name:>22} median {med * 1000:7.1f} ms  p95 {res['latency_p95_s'] * 1000:7.1f} ms  "
          f"batch64 {res['throughput_batch64_tok_s']:7.0f} tok/s  mem +{mem.extra_mb:6.0f} MB  NFE {nfe}", flush=True)
    return res


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.set_num_threads(os.cpu_count())
    P = np.load(os.path.join(DATA, "test_prompt.npy")).astype(np.int64)[:, :16].tolist()
    out = {"threads": os.cpu_count(), "task": "16-token prompt, 48 new tokens", "systems": {}}
    model, ts, _ = load_flow(os.path.join(CHECKPOINTS, "main", "best.pt"))
    pf = nparams(model)
    for solver, steps in (("euler", 8), ("euler", 16), ("euler", 32), ("euler", 64), ("heun", 16), ("heun", 32)):
        nfe = steps * (2 if solver == "heun" else 1)
        out["systems"][f"flow_{solver}{steps}"] = bench(f"flow {solver}{steps}", gen_fn("flow", model, ts, steps, solver),
                                                        P, nfe, pf, 64 * nfe)
    for key, setting in (("ar5", "topp0.95"), ("ar13", "topp0.95")):
        m = load_baseline(key)
        out["systems"][f"{key}_{setting}"] = bench(f"{key} {setting}", gen_fn("ar", m, setting=setting), P, N_NEW,
                                                   nparams(m), 16 + N_NEW)
    for key, setting, steps in (("mdlm5", "small-S32", 32), ("mdlm5", "small-S8", 8), ("mdlm13", "rand-S32", 32)):
        m = load_baseline(key)
        out["systems"][f"{key}_{setting}"] = bench(f"{key} {setting}", gen_fn("diffusion", m, setting=setting), P, steps,
                                                   nparams(m), 65 * steps)
    save_json(out, METRICS, "benchmark.json")


if __name__ == "__main__":
    main()
