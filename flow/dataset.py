"""Fixed-length story sequences (Phase 2).

    python -m flow.dataset

Source: Project 1's token streams (data/full/{train,val,test}.bin, <BOS> story <EOS> back to back, uint16), the same
cleaned TinyStories, tokenizer and splits that the Project 1/2/3 models were trained and tested on. The test split
was never used for training by any of those models, and is not used here for any choice.

Each sequence = the first L tokens after a story's <BOS> (L = 64). If the story ends earlier, the sequence holds the
story, its <EOS>, then <PAD>; valid positions = story tokens + <EOS>. Starting at a story start means every sequence is
a well-formed story opening, so unconditional samples can be judged as story openings and prompted samples continue
a real opening.

Splits written to data/ (uint16 arrays, shape (N, 64)):
  train      every training story                         (flow training)
  val        2,000 validation stories                      (validation MSE, checkpoint choice)
  dev        100 other validation stories with >= 64 tokens (sampling choices: steps, solver, decoder)
  test       2,000 test stories                            (final flow MSE / decoding diagnostics)
  test_prompt 300 other test stories with >= 64 tokens       (common prompts: 16-token prefix, 48-token continuation)
Shorter lengths (L = 32 ablation) use the first 32 columns. Writes data_manifests/data.json.
"""
from __future__ import annotations

import os

import numpy as np

from . import BOS_ID, DATA, EOS_ID, P1_DATA, PAD_ID, PROJECT10, save_json, sha256

L = 64
SEED = 10


def story_starts(split: str):
    toks = np.memmap(os.path.join(P1_DATA, f"{split}.bin"), dtype=np.uint16, mode="r")
    bos = np.flatnonzero(toks == BOS_ID)
    return toks, bos


def build(toks, bos, idx, length=L):
    out = np.full((len(idx), length), PAD_ID, dtype=np.uint16)
    full = np.zeros(len(idx), dtype=bool)
    ends = np.append(bos[1:], len(toks))
    for r, i in enumerate(idx):
        s, e = bos[i] + 1, ends[i]          # tokens after <BOS> up to the next story (includes this story's <EOS>)
        seg = np.asarray(toks[s:min(e, s + length)])
        out[r, :len(seg)] = seg
        full[r] = (e - s) > length          # the story continues beyond the window (no EOS inside)
    return out, full


def valid_mask(seq: np.ndarray) -> np.ndarray:
    """True for story tokens and the first <EOS>; False for <PAD> after it."""
    is_eos = seq == EOS_ID
    after = np.cumsum(is_eos, axis=1) - is_eos   # number of EOS strictly before this position
    return after == 0


def main():
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(SEED)
    man = {"source": P1_DATA, "length": L, "splits": {}, "sha256": {}}
    for split in ("train", "val", "test"):
        man["sha256"][f"project1/data/full/{split}.bin"] = sha256(os.path.join(P1_DATA, f"{split}.bin"))
    # train: all stories, in file order (the trainer shuffles)
    toks, bos = story_starts("train")
    tr, trf = build(toks, bos, np.arange(len(bos)))
    np.save(os.path.join(DATA, "train.npy"), tr)
    man["splits"]["train"] = {"n": len(tr), "full_window_share": float(trf.mean()),
                              "valid_tokens": int(valid_mask(tr).sum())}
    for split, names in (("val", ("val", "dev")), ("test", ("test", "test_prompt"))):
        toks, bos = story_starts(split)
        order = rng.permutation(len(bos))
        seq, full = build(toks, bos, order)
        a = seq[:2000]
        rest = [i for i in range(2000, len(order)) if full[i]]
        n_b = 100 if names[1] == "dev" else 300
        b = seq[rest[:n_b]]
        np.save(os.path.join(DATA, f"{names[0]}.npy"), a)
        np.save(os.path.join(DATA, f"{names[1]}.npy"), b)
        man["splits"][names[0]] = {"n": len(a), "full_window_share": float(full[:2000].mean()), "stories_in_split": len(bos)}
        man["splits"][names[1]] = {"n": len(b), "rule": "next stories (random order) with more than 64 tokens"}
    # overlap check: no identical sequence across splits
    keys = {k: {r.tobytes() for r in np.load(os.path.join(DATA, f"{k}.npy"))} for k in ("val", "dev", "test", "test_prompt")}
    trk = None
    overlaps = {}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}-{b}"] = len(keys[a] & keys[b])
    trk = {r.tobytes() for r in tr}
    for k in names:
        overlaps[f"train-{k}"] = len(trk & keys[k])
    man["identical_64token_openings_across_splits"] = overlaps
    man["note"] = ("identical openings across splits are possible because many TinyStories start the same way; they "
                   "are counted, not removed (the stories themselves are different)")
    lens = valid_mask(tr).sum(1)
    man["train_valid_length_share_lt64"] = float((lens < L).mean())
    save_json(man, PROJECT10, "data_manifests", "data.json")
    for k, v in man["splits"].items():
        print(k, v)
    print("overlaps", overlaps)


if __name__ == "__main__":
    main()
