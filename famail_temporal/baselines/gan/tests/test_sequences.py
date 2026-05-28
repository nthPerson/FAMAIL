"""Unit tests for gan.sequences."""
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan import sequences as sq
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour


def _traj():
    # start at cell (2,3) in hour 8 (time_bucket 8*12+1=97); ends at (5,7)
    states = [
        TrajectoryState(x_grid=2.0, y_grid=3.0, time_bucket=97, day_index=1),
        TrajectoryState(x_grid=4.0, y_grid=6.0, time_bucket=97, day_index=1),
        TrajectoryState(x_grid=5.0, y_grid=7.0, time_bucket=97, day_index=1),
    ]
    return Trajectory(trajectory_id=0, driver_id=0, states=states)


def test_flat_cell_round_trip():
    for (x, y) in [(0, 0), (2, 3), (47, 89)]:
        assert sq.unflat_cell(sq.flat_cell(x, y)) == (x, y)


def test_trajectory_to_tokens_brackets_with_bos_eos():
    toks = sq.trajectory_to_tokens(_traj())
    assert toks[0] == gc.BOS
    assert toks[-1] == gc.EOS
    # Interior tokens are the three states' flat cells.
    assert toks[1:-1] == [sq.flat_cell(2, 3), sq.flat_cell(4, 6), sq.flat_cell(5, 7)]
    assert all(0 <= t < gc.VOCAB_SIZE for t in toks)


def test_trajectory_context_is_start_cell_and_block():
    cell, tblock = sq.trajectory_context(_traj())
    assert cell == sq.flat_cell(2, 3)
    assert tblock == hour_to_block_index(time_bucket_to_hour(97))
