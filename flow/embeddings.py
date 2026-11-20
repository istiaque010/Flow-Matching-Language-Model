"""Continuous token representation and decoders (Phases 3 and 9).

    python -m flow.embeddings          # statistics + round-trip and perturbation tests -> results/metrics/embeddings.json

Representation: Project 1's `small` model token embedding table E (8,192 x 384, trained with its tied output head on
164M tokens), FROZEN. The flow works in a standardised space
    x1 = (E[id] - mu) / sigma
where mu (384-vector) and sigma (scalar) are the token-frequency-weighted mean and standard deviation of E over a
10M-token sample of the training data, so the data the flow sees has mean 0 and variance 1 per dimension on average,
like the Gaussian noise x0. (Raw E has std ~0.04 and non-zero means, far from N(0, I).)

Variant "pca128" (embedding-dimension ablation): the top-128 principal directions of the standardised table
(frequency-weighted), rescaled to unit variance.

Decoders (vector -> token id):
  nearest     the token whose standardised embedding is closest in Euclidean distance (a token-level search over the
              8,192-row table, not a retrieval system)
  projection  Project 1's tied vocabulary head applied to the un-standardised vector: argmax_k (sigma x + mu) . E_k
              (without Project 1's final layer norm, which belongs to its transformer, not to the embedding)
"""
from __future__ import annotations

import os

import numpy as np
import torch

from . import DATA, EMB_CKPT, METRICS, P1_DATA, save_json, sha256

STATS = os.path.join(DATA, "embedding_stats.pt")


def _freq(n_tokens=10_000_000):
    toks = np.memmap(os.path.join(P1_DATA, "train.bin"), dtype=np.uint16, mode="r")
    c = np.bincount(np.asarray(toks[:n_tokens]), minlength=8192).astype(np.float64)
    return torch.tensor(c / c.sum(), dtype=torch.float32)


def compute_stats():
    E = torch.load(EMB_CKPT, map_location="cpu", weights_only=False)["model"]["tok_emb.weight"].float()
    w = _freq()
    mu = (w[:, None] * E).sum(0)
    var = (w[:, None] * (E - mu) ** 2).sum(0)          # per-dimension, frequency weighted
    sigma = var.mean().sqrt()
    Z = (E - mu) / sigma
    # PCA (frequency weighted) for the 128-d variant
    C = (Z * w[:, None]).T @ Z
    evals, evecs = torch.linalg.eigh(C)
    P = evecs[:, evals.argsort(descending=True)[:128]]
    Z128 = Z @ P
    s128 = ((w[:, None] * Z128 ** 2).sum(0).mean()).sqrt()
    st = {"E": E, "mu": mu, "sigma": sigma, "freq": w, "pca": P, "pca_scale": s128,
          "pca_explained": float(evals.sort(descending=True).values[:128].sum() / evals.sum()),
          "emb_sha256": sha256(EMB_CKPT)}
    os.makedirs(DATA, exist_ok=True)
    torch.save(st, STATS)
    return st


class TokenSpace:
    """Maps token ids <-> vectors in the flow's space and decodes vectors back to ids."""

    def __init__(self, kind: str = "p1small"):
        st = torch.load(STATS, weights_only=False) if os.path.exists(STATS) else compute_stats()
        self.kind = kind
        self.E, self.mu, self.sigma, self.freq = st["E"], st["mu"], st["sigma"], st["freq"]
        Z = (self.E - self.mu) / self.sigma
        if kind == "p1small":
            self.table = Z
        elif kind == "pca128":
            self.P, self.pscale = st["pca"], st["pca_scale"]
            self.table = (Z @ self.P) / self.pscale
        else:
            raise ValueError(kind)
        self.dim = self.table.size(1)
        self.table_sq = (self.table ** 2).sum(1)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table[ids.long()]

    @torch.no_grad()
    def nearest(self, x: torch.Tensor, return_dist: bool = False):
        """x (..., d) -> ids (...) by Euclidean nearest neighbour; optionally (d1, d2) = distances to the closest and
        second-closest token (their gap is the decoding margin)."""
        shp = x.shape[:-1]
        flat = x.reshape(-1, self.dim)
        out, d1s, d2s = [], [], []
        for i in range(0, flat.size(0), 4096):
            xb = flat[i:i + 4096]
            d2 = (xb ** 2).sum(1, keepdim=True) - 2 * xb @ self.table.T + self.table_sq[None]
            if return_dist:
                v, ix = d2.topk(2, dim=1, largest=False)
                out.append(ix[:, 0])
                d1s.append(v[:, 0].clamp_min(0).sqrt())
                d2s.append(v[:, 1].clamp_min(0).sqrt())
            else:
                out.append(d2.argmin(1))
        ids = torch.cat(out).view(shp)
        if return_dist:
            return ids, torch.cat(d1s).view(shp), torch.cat(d2s).view(shp)
        return ids

    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Tied vocabulary head (only defined for the full 384-d space)."""
        if self.kind != "p1small":
            raise ValueError("projection decoder needs the 384-d space")
        return ((x * self.sigma + self.mu) @ self.E.T).argmax(-1)

    def decode(self, x, method="nearest"):
        return self.nearest(x) if method == "nearest" else self.project(x)


def main():
    st = compute_stats()
    val = torch.from_numpy(np.load(os.path.join(DATA, "val.npy")).astype(np.int64))
    from .dataset import valid_mask
    m = torch.from_numpy(valid_mask(val.numpy()))
    ids = val[m]
    out = {"emb_ckpt": EMB_CKPT, "emb_sha256": st["emb_sha256"], "raw_std": float(st["E"].std()),
           "sigma": float(st["sigma"]), "mu_norm": float(st["mu"].norm()), "pca128_explained_variance": st["pca_explained"],
           "spaces": {}}
    g = torch.Generator().manual_seed(0)
    for kind in ("p1small", "pca128"):
        ts = TokenSpace(kind)
        x = ts.encode(ids)
        Zt = ts.table
        nn_ids, d1, d2 = ts.nearest(Zt, return_dist=True)   # d1 = 0 (itself), d2 = distance to the nearest other token
        r = {"dim": ts.dim, "data_mean_abs": float(x.mean(0).abs().mean()), "data_std": float(x.std()),
             "vocab_roundtrip_nearest": float((nn_ids == torch.arange(8192)).float().mean()),
             "nearest_other_token_dist_median_vocab": float(d2.median()),
             "nearest_other_token_dist_median_data_weighted": float(d2[ids].median()),
             "clean_val_token_acc": {"nearest": float((ts.nearest(x) == ids).float().mean())}}
        if kind == "p1small":
            r["vocab_roundtrip_projection"] = float((ts.project(Zt) == torch.arange(8192)).float().mean())
            r["clean_val_token_acc"]["projection"] = float((ts.project(x) == ids).float().mean())
        # perturbation: Gaussian noise of std s per dimension added to clean data embeddings (Phase 10 diagnostic)
        r["perturbed_acc"] = {}
        for s in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            xn = x + s * torch.randn(x.shape, generator=g)
            e = {"nearest": float((ts.nearest(xn) == ids).float().mean())}
            if kind == "p1small":
                e["projection"] = float((ts.project(xn) == ids).float().mean())
            r["perturbed_acc"][str(s)] = e
        out["spaces"][kind] = r
    save_json(out, METRICS, "embeddings.json")
    import json
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
