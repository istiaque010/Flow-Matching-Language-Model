"""Paired comparisons of generation runs (Phases 12-14).

    python -m flow.compare            # -> results/metrics/comparisons.json

Unit = prompt (the same 300 test prompts in every prompted run). Scorer NLL difference A - B = token-weighted mean NLL
of A minus that of B; 95% interval from 10,000 bootstrap resamples of prompts (fixed seed). Runs with several seeds
are pooled per prompt (sum of NLL and of tokens over seeds). Also reported for rep4 and unknown words (mean per prompt).

Ablation rule (configs/hypotheses.md): an ablation counts only if its interval against reference seed 0 excludes 0
AND its scorer perplexity lies outside the range of the two reference seeds (ref_s0, ref_s1).
"""
from __future__ import annotations

import os

import numpy as np

from . import GENERATIONS, METRICS, load_json, read_jsonl, save_json


def per_prompt(names):
    rows = [read_jsonl(GENERATIONS, f"{n}.jsonl") for n in names]
    nll = np.sum([[r["scorer_nll_sum"] for r in rs] for rs in rows], axis=0)
    tok = np.sum([[r["scorer_tokens"] for r in rs] for rs in rows], axis=0)
    rep = np.mean([[r["rep4"] for r in rs] for rs in rows], axis=0)
    unk = np.mean([[r["unknown_words"] for r in rs] for rs in rows], axis=0)
    return nll, tok, rep, unk


def boot(fn, n, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    b = np.array([fn(i) for i in idx])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def pair(a, b, label=None):
    na, ta, ra, ua = per_prompt(a)
    nb, tb, rb, ub = per_prompt(b)
    n = len(na)
    d = float(na.sum() / ta.sum() - nb.sum() / tb.sum())
    ci = boot(lambda i: na[i].sum() / ta[i].sum() - nb[i].sum() / tb[i].sum(), n)
    return {"a": a, "b": b, "label": label, "nll_diff": d, "nll_ci95": ci,
            "ppl_a": float(np.exp(na.sum() / ta.sum())), "ppl_b": float(np.exp(nb.sum() / tb.sum())),
            "rep4_diff": float((ra - rb).mean()), "rep4_ci95": boot(lambda i: (ra[i] - rb[i]).mean(), n),
            "unknown_diff": float((ua - ub).mean()), "unknown_ci95": boot(lambda i: (ua[i] - ub[i]).mean(), n)}


def main():
    runs = load_json(METRICS, "runs.json")
    ch = load_json(METRICS, "flow_chosen.json")
    main_flow = [f"test_flow_{ch['solver']}{ch['steps']}_{ch['decoder']}", "test_flow_main_s1", "test_flow_main_s2"]
    ar5 = [f"test_ar5_topp0.95_s{s}" for s in range(3)]
    md5 = [f"test_mdlm5_small-S32_s{s}" for s in range(3)]
    out = {"main": [], "sampling": [], "ablations": []}
    comps = [(main_flow, ar5, "flow vs AR-5.3M"), (main_flow, md5, "flow vs MDLM-5.3M"), (md5, ar5, "MDLM-5.3M vs AR-5.3M"),
             (main_flow, ["test_real_shuffled"], "flow vs real words shuffled"),
             (main_flow, ["test_unigram"], "flow vs unigram words"),
             (["test_real_shuffled"], ["test_unigram"], "shuffled real vs unigram")]
    for a, b, lab in comps:
        if all(x in runs for x in a + b):
            out["main"].append(pair(a, b, lab))
    base = f"test_flow_{ch['solver']}{ch['steps']}_nearest"
    for k in runs:
        if k.startswith("test_flow_") and k != base and not k.startswith("test_flow_main") and k in runs:
            out["sampling"].append(pair([k], [base], f"{k} vs chosen"))
    ref = "abl_ref_s0_test"
    if ref in runs:
        lo, hi = sorted([runs["abl_ref_s0_test"]["scorer_ppl"], runs["abl_ref_s1_test"]["scorer_ppl"]])
        for k in runs:
            if k.startswith("abl_abl_") and k.endswith("_test") or k == "abl_ref_s1_test" or k == "abl_main_auxce_test":
                c = pair([k], [ref], k)
                c["ref_seed_ppl_range"] = [lo, hi]
                c["outside_seed_range"] = not (lo <= runs[k]["scorer_ppl"] <= hi)
                c["counts"] = c["outside_seed_range"] and (c["nll_ci95"][1] < 0 or c["nll_ci95"][0] > 0)
                out["ablations"].append(c)
        if "abl_abl_len32_p8c24" in runs:
            c = pair(["abl_abl_len32_p8c24"], ["abl_ref_s0_p8c24"], "len32 vs ref (8-token prompt, 24 tokens)")
            l2 = sorted([runs["abl_ref_s0_p8c24"]["scorer_ppl"], runs["abl_ref_s1_p8c24"]["scorer_ppl"]])
            c["ref_seed_ppl_range"] = l2
            c["outside_seed_range"] = not (l2[0] <= runs["abl_abl_len32_p8c24"]["scorer_ppl"] <= l2[1])
            c["counts"] = c["outside_seed_range"] and (c["nll_ci95"][1] < 0 or c["nll_ci95"][0] > 0)
            out["ablations"].append(c)
    save_json(out, METRICS, "comparisons.json")
    for grp in ("main", "ablations"):
        for c in out[grp]:
            extra = f"  counts={c['counts']}" if "counts" in c else ""
            print(f"{c['label']:>45}  ppl {c['ppl_a']:9.1f} vs {c['ppl_b']:9.1f}  dNLL {c['nll_diff']:+.3f} "
                  f"[{c['nll_ci95'][0]:+.3f},{c['nll_ci95'][1]:+.3f}]{extra}")


if __name__ == "__main__":
    main()
