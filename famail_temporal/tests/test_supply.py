import numpy as np
import pytest
import torch

from famail_temporal.algorithm import supply as sp
from famail_temporal.algorithm.soft_cell_assignment import SoftCellAssignment


def test_state_presence_mass_convention():
    n_hours = np.ones(24, dtype=np.int32)
    assert sp.state_presence_mass(n_hours, 5, 0) == pytest.approx(1.0 / (12 * 1 * 5))


def test_hard_delta_supply_box_and_sign():
    gx, gy, T = 10, 10, 4
    d = sp.hard_delta_supply(
        positions_old=[(5, 5)], positions_new=[(7, 5)],
        t_blocks=[2], masses=[0.1], grid_shape=(gx, gy, T))
    assert d.shape == (gx, gy, T)
    assert d[5, 5, 2] == pytest.approx(0.0)      # overlap of -box(5,5) and +box(7,5): both cover (5..7,3..7)
    assert d[3, 5, 2] == pytest.approx(-0.1)     # only in old box
    assert d[9, 5, 2] == pytest.approx(+0.1)     # only in new box
    assert d[:, :, 0].sum() == pytest.approx(0.0)  # untouched block
    # clipped at edges: total added mass = 0.1 * |box(7,5)| (full 25 here), removed = 0.1 * 25
    assert d[:, :, 2].sum() == pytest.approx(0.0)


def test_hard_delta_supply_edge_clip():
    d = sp.hard_delta_supply([(0, 0)], [(9, 9)], [0], [1.0], (10, 10, 1))
    assert d[:, :, 0].min() == pytest.approx(-1.0)
    # old box at corner has 3x3=9 cells, new box 3x3=9
    assert (d[:, :, 0] < 0).sum() == 9 and (d[:, :, 0] > 0).sum() == 9


def test_soft_delta_supply_matches_hard_at_low_temperature():
    torch.manual_seed(0)
    gx, gy, T = 12, 12, 3
    sa = SoftCellAssignment(grid_dims=(gx, gy))
    sa.temperature.fill_(1e-4)                    # near-one-hot softmax
    loc = torch.tensor([[6.0, 6.0]]); cell = torch.tensor([[6, 6]])
    probs = sa(loc, cell)                         # (1, ns, ns)
    d_soft = sp.soft_delta_supply(probs, cells=[(6, 6)], t_blocks=[1],
                                  masses=[0.2], signs=[+1], grid_shape=(gx, gy, T))
    d_hard = sp.hard_delta_supply([], [(6, 6)], [1], [0.2], (gx, gy, T))
    np.testing.assert_allclose(d_soft.detach().numpy(), d_hard, atol=1e-4)


def test_soft_delta_supply_differentiable():
    gx, gy, T = 12, 12, 2
    sa = SoftCellAssignment(grid_dims=(gx, gy))
    loc = torch.tensor([[6.0, 6.0]], requires_grad=True)
    probs = sa(loc, torch.tensor([[6, 6]]))
    d = sp.soft_delta_supply(probs, [(6, 6)], [0], [0.2], [+1], (gx, gy, T))
    d.sum().backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
