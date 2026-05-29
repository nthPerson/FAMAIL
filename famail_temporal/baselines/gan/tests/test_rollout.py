"""Unit tests for gan.rollout."""
import numpy as np
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan import rollout as rl
from famail_temporal.baselines.datasets import pickup_mass


def test_sample_cells_are_valid_and_bounded():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    cells = rl.sample_trajectory_cells(
        model, ctx_cell=5, ctx_tblock=0,
        max_len=16, device=torch.device("cpu"),
    )
    assert 0 <= len(cells) <= 16
    assert all(0 <= c < gc.N_CELLS for c in cells)  # specials stripped


def test_pickups_to_grid_scale_and_placement():
    bundle = _make_synthetic_bundle()
    # Two pickups at distinct units; grid mass equals sum of pickup masses.
    pickups = [(2, 3, 0), (2, 3, 0), (4, 5, 1)]
    grid = rl.pickups_to_pickup_3d(bundle, pickups)
    assert grid.shape == bundle.pickup_3d.shape
    assert grid[2, 3, 0] == np.float32(2 * pickup_mass(bundle, 0))
    assert grid[4, 5, 1] == np.float32(pickup_mass(bundle, 1))
    # Untouched cells are zero.
    assert grid.sum() > 0
    grid[2, 3, 0] = 0.0
    grid[4, 5, 1] = 0.0
    assert np.allclose(grid, 0.0)


def test_pickups_outside_grid_are_skipped():
    """Pickups beyond the bundle's grid (vocab > grid in tests) are dropped."""
    bundle = _make_synthetic_bundle()
    gx, gy, n_t = bundle.pickup_3d.shape
    pickups = [(gx + 5, gy + 5, 0), (1, 1, 0), (0, 0, n_t + 3)]
    grid = rl.pickups_to_pickup_3d(bundle, pickups)
    # Only the in-bounds (1, 1, 0) pickup is counted.
    assert grid[1, 1, 0] == np.float32(pickup_mass(bundle, 0))
    assert grid.sum() == np.float32(pickup_mass(bundle, 0))


def test_generate_pickups_is_seed_deterministic():
    model = TrajectoryLSTM()
    contexts = [(5, 0), (9, 1), (3, 0)]
    torch.manual_seed(7)
    a = rl.generate_pickups(model, contexts, max_len=16, device=torch.device("cpu"))
    torch.manual_seed(7)
    b = rl.generate_pickups(model, contexts, max_len=16, device=torch.device("cpu"))
    assert a == b
    assert len(a) == len(contexts)
    # Each pickup inherits its context's time-block (Phase-2 simplification).
    assert [p[2] for p in a] == [c[1] for c in contexts]


def test_sample_terminal_cells_batched_valid_and_deterministic():
    model = TrajectoryLSTM()
    cc = torch.tensor([5, 9, 3, 0], dtype=torch.long)
    tb = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    torch.manual_seed(1)
    a_term, a_len = rl.sample_terminal_cells_batched(
        model, cc, tb, max_len=16, device=torch.device("cpu"),
    )
    torch.manual_seed(1)
    b_term, b_len = rl.sample_terminal_cells_batched(
        model, cc, tb, max_len=16, device=torch.device("cpu"),
    )
    assert a_term.shape == (4,) and a_len.shape == (4,)
    assert torch.equal(a_term, b_term) and torch.equal(a_len, b_len)  # deterministic
    assert bool((a_term < gc.N_CELLS).all())         # every terminal is a cell
    assert bool((a_len >= 1).all()) and bool((a_len <= 16).all())  # length bounds


def test_generate_pickups_batches_match_count_across_batch_sizes():
    """gen_batch_size changes throughput, not the contract: one pickup per
    context, each with the context's time-block, terminals in-vocabulary."""
    model = TrajectoryLSTM()
    contexts = [(c, c % 3) for c in range(10)]
    out = rl.generate_pickups(
        model, contexts, max_len=12, device=torch.device("cpu"), gen_batch_size=4,
    )
    assert len(out) == len(contexts)
    assert [p[2] for p in out] == [c[1] for c in contexts]
    assert all(0 <= x < gc.GX and 0 <= y < gc.GY for (x, y, _) in out)
