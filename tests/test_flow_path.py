import torch

import flow  # noqa: F401
from flow.flow_path import build_xt, implied_x1, interpolate, masked_mse, sample_t, target_velocity


def test_boundaries_and_velocity():
    g = torch.Generator().manual_seed(0)
    x0, x1 = torch.randn(3, 5, 4, generator=g), torch.randn(3, 5, 4, generator=g)
    assert torch.allclose(interpolate(x0, x1, torch.zeros(3)), x0)
    assert torch.allclose(interpolate(x0, x1, torch.ones(3)), x1)
    t = torch.tensor([0.2, 0.5, 0.9])
    xt = interpolate(x0, x1, t)
    # finite-difference derivative along the path equals the target velocity
    h = 1e-3
    d = (interpolate(x0, x1, t + h) - xt) / h
    assert torch.allclose(d, target_velocity(x0, x1), atol=1e-3)
    # the true velocity recovers x1 exactly
    assert torch.allclose(implied_x1(xt, target_velocity(x0, x1), t), x1, atol=1e-5)


def test_sample_t_range():
    t = sample_t(10000, torch.Generator().manual_seed(1))
    assert t.min() > 0 and t.max() < 1 and abs(t.mean().item() - 0.5) < 0.02


def test_masked_mse_ignores_masked_positions():
    p, q = torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)
    q[0, 2] = 100.0                      # huge error at a masked position
    q[1, 0] = 1.0                        # error 1 at one valid position
    m = torch.tensor([[True, True, False], [True, True, True]])
    assert abs(masked_mse(p, q, m).item() - 4 * 1.0 / (5 * 4)) < 1e-6


def test_build_xt_keeps_prefix_clean():
    x0, x1 = torch.randn(2, 6, 3), torch.randn(2, 6, 3)
    pm = torch.zeros(2, 6, dtype=torch.bool)
    pm[:, :2] = True
    xt = build_xt(x1, x0, torch.tensor([0.1, 0.7]), pm)
    assert torch.equal(xt[:, :2], x1[:, :2])
    assert not torch.allclose(xt[:, 2:], x1[:, 2:])
