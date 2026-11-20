"""Text-level evaluation shared by every system (Phases 10, 12, 13).

Training losses are NOT compared across model families (AR perplexity, masked-diffusion bound and flow MSE measure
different things). Every system is judged on its generated text with the same measures:

  scorer_ppl      exp(token-weighted mean NLL) of the continuation given the prompt under an independent scorer,
                  TinyStories-33M (GPT-Neo, MIT, revision 2ad0a164; different tokenizer and training run from every
                  model here). Project 4's `scorer_nll` code is reused unchanged. Lower = more fluent / plausible.
  distinct_1/2    distinct words / word pairs over all continuations of a run (corpus level). Higher = more varied.
  rep4            per continuation, share of word 4-grams that already occurred earlier in it (mean over samples).
  dup_adjacent    share of adjacent identical tokens (e.g. "the the").
  unknown_words   share of words not in Project 2's list of 300,000-story known words (made-up or broken words).
  eos             share of continuations that produced <EOS> inside the 48-token window (real openings almost never
                  end there: 0.0% of the 300 test openings do).
  real reference  the true continuation (tokens 16-63) of each test opening is scored the same way.

    python -m flow.evaluate --file results/generations/<name>.jsonl      # adds scores to a generation file
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

import numpy as np

from . import GENERATIONS, METRICS, save_json

WORD = re.compile(r"[A-Za-z']+")
_SC = {}


def scorer():
    if not _SC:
        from bench.load_models import load_scorer
        _SC["m"], _SC["t"] = load_scorer()
    return _SC["m"], _SC["t"]


def known():
    from bench.evaluate_quality import known_words
    return known_words()


def rep4(words):
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    seen, rep = set(), 0
    for gm in grams:
        rep += gm in seen
        seen.add(gm)
    return rep / len(grams) if grams else 0.0


def text_measures(tok, rows):
    """Adds prompt/continuation text and per-sample measures in place; returns corpus-level measures."""
    kw = known()
    allw = []
    for r in rows:
        r["prompt"] = tok.decode(r["prompt_ids"]) if r["prompt_ids"] else ""
        r["text"] = tok.decode(r["cont_ids"])
        w = WORD.findall(r["text"].lower())
        allw.append(w)
        r["n_words"] = len(w)
        r["rep4"] = rep4(w)
        ids = r["cont_ids"]
        r["dup_adjacent"] = sum(a == b for a, b in zip(ids, ids[1:])) / max(1, len(ids) - 1)
        r["unknown_words"] = sum(x not in kw for x in w) / max(1, len(w))
    flat = [x for w in allw for x in w]
    pairs = [tuple(w[i:i + 2]) for w in allw for i in range(len(w) - 1)]
    return {"distinct_1": len(set(flat)) / max(1, len(flat)), "distinct_2": len(set(pairs)) / max(1, len(pairs))}


def score(rows, batch=32):
    m, st = scorer()
    from bench.evaluate_quality import scorer_nll
    pairs = []
    for r in rows:
        p = r["prompt"]
        c = r["text"]
        if p and c and not c.startswith((" ", ".", ",", "!", "?", "'", '"')):
            c = " " + c
        pairs.append((p, c))
    res = scorer_nll(m, st, pairs, batch)
    for r, (nll, n) in zip(rows, res):
        r["scorer_nll_sum"], r["scorer_tokens"] = nll, n


def summarize(rows, corpus):
    nll = sum(r["scorer_nll_sum"] for r in rows)
    n = sum(r["scorer_tokens"] for r in rows)
    out = {"n": len(rows), "scorer_ppl": float(np.exp(nll / max(1, n))), "scorer_nll": nll / max(1, n), **corpus}
    for k in ("rep4", "dup_adjacent", "unknown_words", "n_words"):
        out[k] = float(np.mean([r[k] for r in rows]))
    out["eos"] = float(np.mean([r["ended"] for r in rows]))
    out["repetitive_share"] = float(np.mean([r["rep4"] > 0.1 for r in rows]))
    if "dist_nearest" in rows[0]:
        out["dist_nearest"] = float(np.mean([r["dist_nearest"] for r in rows]))
        out["margin"] = float(np.mean([r["margin"] for r in rows]))
    out["nfe"] = float(np.mean([r["nfe"] for r in rows]))
    return out


def evaluate_rows(rows, tok):
    corpus = text_measures(tok, rows)
    score(rows)
    return summarize(rows, corpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    a = ap.parse_args()
    from . import load_tokenizer
    rows = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    s = evaluate_rows(rows, load_tokenizer())
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
