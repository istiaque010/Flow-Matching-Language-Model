import numpy as np
import torch

import flow  # noqa: F401
from flow.dataset import valid_mask
from flow.model import FlowConfig, VelocityTransformer


def _model(time_cond="adaln"):
    torch.manual_seed(0)
    m = VelocityTransformer(FlowConfig(d_data=12, seq_len=8, n_layer=2, n_head=2, width=16, time_cond=time_cond))
    # zero-initialised gates/output make the fresh model output 0 for every t; randomise them for the response test
    for p in m.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    return m.eval()


def test_output_shape_and_time_response():
    m = _model()
    x = torch.randn(3, 8, 12)
    v1 = m(x, torch.full((3,), 0.1))
    v2 = m(x, torch.full((3,), 0.9))
    assert v1.shape == (3, 8, 12)
    assert (v1 - v2).abs().max() > 1e-3


def test_no_time_ablation_ignores_t():
    m = _model("none")
    x = torch.randn(2, 8, 12)
    assert torch.allclose(m(x, torch.full((2,), 0.1)), m(x, torch.full((2,), 0.9)))


def test_fresh_model_predicts_zero():
    m = VelocityTransformer(FlowConfig(d_data=12, seq_len=8, n_layer=2, n_head=2, width=16))
    assert m(torch.randn(2, 8, 12), torch.rand(2)).abs().max() == 0


def test_padding_mask_blocks_attention():
    m = _model()
    x = torch.randn(1, 8, 12)
    kp = torch.ones(1, 8, dtype=torch.bool)
    kp[0, 6:] = False
    y = x.clone()
    y[0, 6:] = 50.0                      # change only padded positions
    a = m(x, torch.tensor([0.5]), None, kp)[0, :6]
    b = m(y, torch.tensor([0.5]), None, kp)[0, :6]
    assert torch.allclose(a, b, atol=1e-5)


def test_valid_mask():
    seq = np.array([[5, 6, 3, 0, 0], [5, 6, 7, 8, 9]])
    assert valid_mask(seq).tolist() == [[True, True, True, False, False], [True] * 5]
