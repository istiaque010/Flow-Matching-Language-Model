import torch

import flow  # noqa: F401
from flow.ode_sampler import integrate


def test_constant_velocity_exact_and_nfe():
    x0 = torch.zeros(2, 3, 1)
    for solver, steps, nfe in (("euler", 8, 8), ("heun", 8, 16), ("euler", 1, 1)):
        x, n, _ = integrate(lambda x, t: torch.ones_like(x) * 2.0, x0, steps, solver)
        assert torch.allclose(x, torch.full_like(x0, 2.0)) and n == nfe


def test_heun_second_order_on_time_dependent_field():
    # dx/dt = t  ->  x(1) = 0.5. Euler with N steps gives (N - 1) / (2N); Heun is exact for a linear field in t.
    f = lambda x, t: t.view(-1, 1, 1).expand_as(x)  # noqa: E731
    x0 = torch.zeros(1, 1, 1)
    xe, _, _ = integrate(f, x0, 4, "euler")
    xh, _, _ = integrate(f, x0, 4, "heun")
    assert abs(xe.item() - 3 / 8) < 1e-6
    assert abs(xh.item() - 0.5) < 1e-6


def test_clamp_keeps_prefix_and_path_length():
    x0 = torch.randn(2, 4, 3)
    m = torch.zeros(2, 4, dtype=torch.bool)
    m[:, :1] = True
    vals = torch.full((2, 4, 3), 7.0)
    x, _, path = integrate(lambda x, t: torch.ones_like(x), x0, 5, "heun", clamp=(m, vals), return_path=True)
    assert torch.equal(x[:, 0], vals[:, 0]) and len(path) == 6
    assert all(torch.equal(p[:, 0], vals[:, 0]) for p in path)
