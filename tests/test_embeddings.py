import torch

import flow  # noqa: F401
from flow.embeddings import TokenSpace


def test_roundtrip_and_standardisation():
    ts = TokenSpace("p1small")
    ids = torch.arange(8192)
    assert torch.equal(ts.nearest(ts.encode(ids)), ids)
    # frequency-weighted data statistics are ~standard normal per dimension
    w = ts.freq[:, None]
    mean = (w * ts.table).sum(0)
    var = (w * (ts.table - mean) ** 2).sum(0).mean()
    assert mean.abs().max() < 1e-3 and abs(var.item() - 1) < 1e-3


def test_projection_matches_tied_head_on_frequent_tokens():
    ts = TokenSpace("p1small")
    top = ts.freq.argsort(descending=True)[:500]
    assert (ts.project(ts.encode(top)) == top).float().mean() > 0.98


def test_pca_space_dims():
    ts = TokenSpace("pca128")
    assert ts.encode(torch.tensor([[5, 6]])).shape == (1, 2, 128)
