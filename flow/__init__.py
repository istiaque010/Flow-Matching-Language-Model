"""Project 10: a flow-matching language model on TinyStories (package `flow`).

Importing `flow` makes Project 4's package `bench` importable (model loaders, KV-cached AR generator, diffusion
generator wrapper, scorer loader), which in turn makes Project 3's `mdlm` and Project 1's `src` importable. Projects
1-4 are only read, never changed; bytecode caching is switched off so nothing is written into their folders.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True

PROJECT10 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PROJECT10)
PROJECT1 = os.path.join(ROOT, "project1")
PROJECT2 = os.path.join(ROOT, "project2")
PROJECT3 = os.path.join(ROOT, "project3")
PROJECT4 = os.path.join(ROOT, "project4")
CONFIGS = os.path.join(PROJECT10, "configs")
DATA = os.path.join(PROJECT10, "data")
CHECKPOINTS = os.path.join(PROJECT10, "checkpoints")
RESULTS = os.path.join(PROJECT10, "results")
METRICS = os.path.join(RESULTS, "metrics")
GENERATIONS = os.path.join(RESULTS, "generations")
FIGURES = os.path.join(RESULTS, "figures")

P1_DATA = os.path.join(PROJECT1, "data", "full")          # <BOS> story <EOS> streams, uint16
TOKENIZER = os.path.join(PROJECT1, "data", "tokenizer.json")
EMB_CKPT = os.path.join(PROJECT1, "checkpoints", "small", "model_best.pt")   # frozen embedding + tied head
SCORER_DIR = os.path.join(PROJECT4, "checkpoints", "scorer_tinystories33m")

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

for p in (PROJECT4, PROJECT3, PROJECT1):
    if p not in sys.path:
        sys.path.insert(0, p)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    return p


def load_json(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def read_jsonl(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_config(name: str) -> dict:
    import yaml
    with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer():
    from src.tokenizer import StoryTokenizer
    return StoryTokenizer(TOKENIZER)
