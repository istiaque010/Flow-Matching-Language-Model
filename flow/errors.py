"""Error analysis and failure taxonomy (Phase 15).

    python -m flow.errors --stage auto      # automatic measures per system -> results/metrics/errors_auto.json
    python -m flow.errors --stage sample    # blind review sample (systems hidden, order shuffled)
    (label results/error_review/blind.jsonl -> results/error_review/labels_ai.json; AI labels, disclosed as such)
    python -m flow.errors --stage count     # un-blind and count the review labels -> results/metrics/errors_review.json

Automatic measures over the 300 test continuations of each system (seed 0):
  bigram_seen     share of adjacent token pairs (prompt end -> continuation included) that occur at least once in
                  20M training tokens. Real text is near 1; a bag of words is far lower. A cheap "local syntax" test.
  name_kept       of prompts that introduce a name ("named Ben"), the share of continuations that use that name again
  prompt_overlap  share of the continuation's distinct words (lower case, letters only) that also appear in the prompt
  dup_adjacent    share of continuations with the same word twice in a row
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import re

import numpy as np

from . import DATA, GENERATIONS, METRICS, RESULTS, load_json, read_jsonl, save_json, write_jsonl

NAMED = re.compile(r"\bnamed ([A-Z][a-z]+)")
WORD = re.compile(r"[a-z]+")
REVIEW = os.path.join(RESULTS, "error_review")
N_REVIEW = 20
AUTO_SYSTEMS = {"real": "test_real", "real_shuffled": "test_real_shuffled", "unigram": "test_unigram",
                "ar5": "test_ar5_topp0.95_s0", "mdlm5": "test_mdlm5_small-S32_s0", "ar13": "test_ar13_topp0.95_s0",
                "mdlm13": "test_mdlm13_rand-S32_s0"}


def flow_name():
    ch = load_json(METRICS, "flow_chosen.json")
    return f"test_flow_{ch['solver']}{ch['steps']}_{ch['decoder']}"


def bigram_set(n_tokens=20_000_000):
    a = np.load(os.path.join(DATA, "train.npy"), mmap_mode="r")
    rows = n_tokens // a.shape[1]
    x = np.asarray(a[:rows]).astype(np.int64)
    pairs = (x[:, :-1] * 8192 + x[:, 1:]).reshape(-1)
    return set(np.unique(pairs).tolist())


def measures(rows, seen):
    bs, kept, ov, dup = [], [], [], []
    for r in rows:
        seq = r["prompt_ids"][-1:] + r["cont_ids"]
        if len(seq) > 1:
            bs.append(np.mean([a * 8192 + b in seen for a, b in zip(seq[:-1], seq[1:])]))
        names = NAMED.findall(r["prompt"])
        if names:
            kept.append(any(n in r["text"] for n in names))
        cw = set(WORD.findall(r["text"].lower()))
        pw = set(WORD.findall(r["prompt"].lower()))
        ov.append(len(cw & pw) / max(1, len(cw)))
        dup.append(r.get("dup_adjacent", 0) > 0)
    return {"bigram_seen": float(np.mean(bs)), "name_kept": float(np.mean(kept)) if kept else None,
            "named_prompts": len(kept), "prompt_overlap": float(np.mean(ov)), "dup_adjacent": float(np.mean(dup))}


def stage_auto():
    seen = bigram_set()
    systems = {"flow": flow_name(), **AUTO_SYSTEMS}
    runs = load_json(METRICS, "runs.json")
    for k in ("abl_abl_tpow2_test", "abl_abl_auxce_test", "abl_main_auxce_test", "abl_ref_s0_test"):
        if k in runs:
            systems[k.replace("abl_", "", 1).replace("_test", "")] = k
    out = {"note": f"bigram set from 20M training tokens ({len(seen):,} distinct pairs)"}
    for s, f in systems.items():
        if f in runs:
            out[s] = measures(read_jsonl(GENERATIONS, f"{f}.jsonl"), seen)
            print(f"{s:>14}", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in out[s].items()})
    save_json(out, METRICS, "errors_auto.json")


def review_systems():
    return {"flow": flow_name(), "ar5": "test_ar5_topp0.95_s0", "mdlm5": "test_mdlm5_small-S32_s0", "real": "test_real"}


def stage_sample():
    rng = random.Random(77)
    pick = sorted(rng.sample(range(300), N_REVIEW))
    items = []
    for s, f in review_systems().items():
        rows = read_jsonl(GENERATIONS, f"{f}.jsonl")
        for i in pick:
            items.append({"system": s, "prompt": rows[i]["prompt"], "continuation": rows[i]["text"], "prompt_index": i})
    rng.shuffle(items)
    blind = [{"n": k, "prompt": it["prompt"], "continuation": it["continuation"]} for k, it in enumerate(items)]
    key = [{"n": k, "system": it["system"], "prompt_index": it["prompt_index"]} for k, it in enumerate(items)]
    write_jsonl(blind, REVIEW, "blind.jsonl")
    write_jsonl(key, REVIEW, "key.jsonl")
    print(f"{len(blind)} blind items -> {REVIEW}")


def stage_count():
    key = {r["n"]: r for r in read_jsonl(REVIEW, "key.jsonl")}
    lab = load_json(REVIEW, "labels_ai.json")["labels"]
    by = collections.defaultdict(list)
    for l in lab:
        by[key[l["n"]]["system"]].append(l)
    out = {}
    for s, ls in by.items():
        issues = collections.Counter(i for l in ls for i in l["issues"] if i != "none")
        out[s] = {"n": len(ls), "coherence_mean": float(np.mean([l["coherence"] for l in ls])),
                  "relevance_mean": float(np.mean([l["relevance"] for l in ls])),
                  "grammatical_share": float(np.mean([l["coherence"] >= 1 for l in ls])),
                  "issues": dict(issues.most_common())}
    save_json({"annotator": "AI (Claude subagent), blind to system; not human", "systems": out},
              METRICS, "errors_review.json")
    for s, v in out.items():
        print(f"{s:>8} coherence {v['coherence_mean']:.2f} relevance {v['relevance_mean']:.2f} {v['issues']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["auto", "sample", "count"], required=True)
    a = ap.parse_args()
    {"auto": stage_auto, "sample": stage_sample, "count": stage_count}[a.stage]()


if __name__ == "__main__":
    main()
