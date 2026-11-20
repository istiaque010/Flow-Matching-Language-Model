"""Generate and score every system (Phases 9, 11, 12, 13, 14).

    python -m flow.run_eval --stage dev        # choose the flow decoding setting on 100 validation prompts
    python -m flow.run_eval --stage test       # all systems on the 300 test prompts (+ unconditional samples)
    python -m flow.run_eval --stage ablations  # every ablation checkpoint with the chosen setting

Prompted task: the first 16 tokens of a held-out opening are the prompt, 48 new tokens are generated (all systems get
the same prompt ids and the same budget). Unconditional task (flow and AR only): 64 tokens from nothing (AR: from
<BOS>). References scored the same way:
  real            the true continuation (tokens 16-63) of each opening
  real_shuffled   the same tokens in a random order (a "right words, no syntax" reference)
  unigram         48 tokens drawn independently from the training-token frequencies (no context at all)
Each run is written to results/generations/<name>.jsonl and summarised in results/metrics/runs.json.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from . import CHECKPOINTS, DATA, GENERATIONS, METRICS, load_json, load_tokenizer, save_json, write_jsonl
from .evaluate import evaluate_rows
from .generate import ar_sample, diffusion_sample, flow_sample, load_baseline, load_flow

K_PROMPT, N_NEW = 16, 48
SWEEP = [("euler", 8), ("euler", 16), ("euler", 32), ("euler", 64), ("heun", 4), ("heun", 8), ("heun", 16), ("heun", 32)]
RUNS = os.path.join(METRICS, "runs.json")


def prompts_of(split):
    a = np.load(os.path.join(DATA, f"{split}.npy")).astype(np.int64)
    return a[:, :K_PROMPT].tolist(), a[:, K_PROMPT:].tolist()


def record(name, rows, tok, meta):
    for r in rows:
        r.update(meta)
    s = evaluate_rows(rows, tok)
    s.update(meta)
    write_jsonl(rows, GENERATIONS, f"{name}.jsonl")
    runs = load_json(RUNS) if os.path.exists(RUNS) else {}
    runs[name] = s
    save_json(runs, RUNS)
    print(f"{name:>34}  ppl {s['scorer_ppl']:8.2f}  d2 {s['distinct_2']:.3f}  rep4 {s['rep4']:.3f} "
          f"unk {s['unknown_words']:.3f}  eos {s['eos']:.2f}  nfe {s['nfe']:.0f}", flush=True)
    return s


def flow_batched(model, ts, prompts, steps, solver, decoder, seed, uncond_n=None, bs=100):
    out = []
    if prompts is None:
        for i in range(0, uncond_n, bs):
            r, _ = flow_sample(model, ts, min(bs, uncond_n - i), steps, solver, None, seed * 1000 + i, decoder)
            out += r
        return out
    for i in range(0, len(prompts), bs):
        r, _ = flow_sample(model, ts, len(prompts[i:i + bs]), steps, solver, prompts[i:i + bs], seed * 1000 + i, decoder)
        for x in r:
            x["cont_ids"] = x["cont_ids"][:N_NEW]
        out += r
    return out


def stage_dev(tok):
    model, ts, _ = load_flow(os.path.join(CHECKPOINTS, "main", "best.pt"))
    prompts, _ = prompts_of("dev")
    res = {}
    for solver, steps in SWEEP:
        for dec in ("nearest", "projection"):
            name = f"dev_flow_{solver}{steps}_{dec}"
            res[name] = record(name, flow_batched(model, ts, prompts, steps, solver, dec, 0), tok,
                               {"system": "flow", "solver": solver, "steps": steps, "decoder": dec, "split": "dev"})
    best = min(res, key=lambda k: (round(res[k]["scorer_ppl"], 2), res[k]["nfe"]))
    chosen = {"solver": res[best]["solver"], "steps": res[best]["steps"], "decoder": res[best]["decoder"],
              "rule": "lowest dev scorer perplexity, ties -> fewer NFE", "dev_ppl": res[best]["scorer_ppl"]}
    save_json(chosen, METRICS, "flow_chosen.json")
    print("chosen", chosen)


def references(tok, prompts, conts, seed=0):
    rng = random.Random(seed)
    real = [{"prompt_ids": p, "cont_ids": c, "ended": False, "nfe": 0} for p, c in zip(prompts, conts)]
    shuf = []
    for p, c in zip(prompts, conts):
        c2 = list(c)
        rng.shuffle(c2)
        shuf.append({"prompt_ids": p, "cont_ids": c2, "ended": False, "nfe": 0})
    from .embeddings import TokenSpace
    freq = TokenSpace("p1small").freq.numpy().astype(np.float64)
    freq[[0, 1, 2, 3]] = 0
    freq /= freq.sum()
    g = np.random.default_rng(seed)
    uni = [{"prompt_ids": p, "cont_ids": g.choice(8192, N_NEW, p=freq).tolist(), "ended": False, "nfe": 0}
           for p in prompts]
    record("test_real", real, tok, {"system": "real", "split": "test"})
    record("test_real_shuffled", shuf, tok, {"system": "real_shuffled", "split": "test"})
    record("test_unigram", uni, tok, {"system": "unigram", "split": "test"})


def stage_test(tok):
    ch = load_json(METRICS, "flow_chosen.json")
    prompts, conts = prompts_of("test_prompt")
    references(tok, prompts, conts)
    model, ts, _ = load_flow(os.path.join(CHECKPOINTS, "main", "best.pt"))
    # step / solver / decoder curves on test (Phase 12), seed 0
    for solver, steps in SWEEP:
        for dec in ("nearest", "projection"):
            record(f"test_flow_{solver}{steps}_{dec}", flow_batched(model, ts, prompts, steps, solver, dec, 0), tok,
                   {"system": "flow", "solver": solver, "steps": steps, "decoder": dec, "seed": 0, "split": "test"})
    for seed in (1, 2):
        record(f"test_flow_main_s{seed}", flow_batched(model, ts, prompts, ch["steps"], ch["solver"], ch["decoder"], seed),
               tok, {"system": "flow", **{k: ch[k] for k in ("solver", "steps", "decoder")}, "seed": seed, "split": "test"})
    # unconditional
    for solver, steps in (("euler", 8), ("euler", 32), (ch["solver"], ch["steps"])):
        name = f"uncond_flow_{solver}{steps}"
        rows = flow_batched(model, ts, None, steps, solver, ch["decoder"], 0, uncond_n=300)
        record(name, rows, tok, {"system": "flow", "solver": solver, "steps": steps, "decoder": ch["decoder"],
                                 "split": "uncond"})
    # baselines
    base = [("ar5", "ar", "topp0.95", (0, 1, 2)), ("ar5", "ar", "temp1.0", (0,)), ("mdlm5", "diffusion", "small-S32", (0, 1, 2)),
            ("mdlm5", "diffusion", "small-S8", (0,)), ("ar13", "ar", "topp0.95", (0,)), ("mdlm13", "diffusion", "rand-S32", (0,))]
    for key, fam, setting, seeds in base:
        m = load_baseline(key)
        for seed in seeds:
            rows = []
            for i in range(0, len(prompts), 100):
                ps = prompts[i:i + 100]
                rows += (ar_sample(m, ps, N_NEW, setting, seed * 1000 + i) if fam == "ar"
                         else diffusion_sample(m, ps, N_NEW, setting, seed * 1000 + i))
            record(f"test_{key}_{setting}_s{seed}", rows, tok, {"system": key, "setting": setting, "seed": seed,
                                                              "split": "test"})
        if key == "ar5":       # unconditional AR: 64 tokens from <BOS>
            from bench.generate_ar import ar_generate_ids
            from . import BOS_ID
            from .generate import AR_SAMPLING
            g = torch.Generator().manual_seed(0)
            out, ended, _ = ar_generate_ids(m, [[BOS_ID]] * 300, 64, AR_SAMPLING["topp0.95"], g)
            record("uncond_ar5_topp0.95", [{"prompt_ids": [], "cont_ids": o, "ended": e, "nfe": len(o)}
                                            for o, e in zip(out, ended)], tok,
                   {"system": "ar5", "setting": "topp0.95", "split": "uncond"})


ABLATIONS = ["ref_s0", "ref_s1", "abl_notime", "abl_auxce", "abl_pca128", "abl_tpow2", "abl_len32"]


def stage_ablations(tok, which=None):
    ch = load_json(METRICS, "flow_chosen.json")
    prompts, _ = prompts_of("test_prompt")
    p8 = [p[:8] for p in prompts]
    for run in (which or ABLATIONS) + (["main_auxce"] if os.path.exists(os.path.join(CHECKPOINTS, "main_auxce")) else []):
        path = os.path.join(CHECKPOINTS, run, "best.pt")
        if not os.path.exists(path):
            continue
        model, ts, _ = load_flow(path)
        meta = {"system": "flow", "run": run, **{k: ch[k] for k in ("solver", "steps", "decoder")}, "seed": 0}
        dec = ch["decoder"] if ts.kind == "p1small" else "nearest"
        if run != "abl_len32":
            record(f"abl_{run}_test", flow_batched(model, ts, prompts, ch["steps"], ch["solver"], dec, 0), tok,
                   {**meta, "split": "test"})
        # length comparison on the common 8-token prompt / 24-token continuation (positions 8-31)
        if run in ("ref_s0", "ref_s1", "abl_len32"):
            rows = []
            for i in range(0, len(p8), 100):
                r, _ = flow_sample(model, ts, len(p8[i:i + 100]), ch["steps"], ch["solver"], p8[i:i + 100], i, dec)
                for x in r:
                    x["cont_ids"] = x["cont_ids"][:24]
                rows += r
            record(f"abl_{run}_p8c24", rows, tok, {**meta, "split": "test_p8c24"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "test", "ablations"], required=True)
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    tok = load_tokenizer()
    if a.stage == "dev":
        stage_dev(tok)
    elif a.stage == "test":
        stage_test(tok)
    else:
        stage_ablations(tok, a.runs)


if __name__ == "__main__":
    main()
