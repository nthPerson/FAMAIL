import torch

from famail_temporal.baselines import run_level2_table as r2
from famail_temporal.baselines.gan import config as gc


class _State:
    def __init__(self, x, y, t=10, d=1):
        self.x_grid, self.y_grid, self.time_bucket, self.day_index = x, y, t, d


class _Traj:
    def __init__(self, tid, did, cells):
        self.trajectory_id, self.driver_id = tid, did
        self.states = [_State(x, y) for (x, y) in cells]


class _Hist:
    def __init__(self, original, modified):
        self.original, self.modified = original, modified


def test_build_edited_corpus_swaps_by_id():
    raw = [_Traj(0, 0, [(0, 0), (1, 1)]), _Traj(1, 0, [(2, 2)]), _Traj(2, 1, [(3, 3)])]
    mod = _Traj(1, 0, [(9, 9)])
    histories = [_Hist(raw[1], mod)]
    edited = r2.build_edited_corpus(raw, histories)
    assert len(edited) == 3
    assert edited[0] is raw[0] and edited[2] is raw[2]      # unchanged kept
    assert edited[1] is mod                                  # modified swapped in
    assert [(s.x_grid, s.y_grid) for s in edited[1].states] == [(9, 9)]


def test_traj_training_data_aligned():
    raw = [_Traj(0, 5, [(0, 0), (1, 1)]), _Traj(1, 7, [(2, 2)])]
    d2i = {5: 0, 7: 1}
    out = r2.traj_training_data(raw, d2i)
    assert len(out["sequences"]) == len(out["contexts"]) == len(out["driver_idxs"]) == 2
    assert out["driver_idxs"] == [0, 1]
    assert out["sequences"][0][0] == gc.BOS and out["sequences"][0][-1] == gc.EOS


def test_gen_training_data_empty_fallback(monkeypatch):
    raw = [_Traj(0, 5, [(4, 4), (5, 5)]), _Traj(1, 7, [(6, 6)])]
    d2i = {5: 0, 7: 1}
    # stub generate_trajectories: first rollout empty, second non-empty
    monkeypatch.setattr(r2, "generate_trajectories", lambda *a, **k: [[], [12]])
    out = r2.gen_training_data(object(), raw, d2i, max_len=8, device=torch.device("cpu"),
                               gen_batch_size=4)
    assert len(out["sequences"]) == 2 and out["n_empty"] == 1
    # empty -> [BOS, start_cell, EOS]; start cell = flat_cell(4,4)
    from famail_temporal.baselines.gan.sequences import flat_cell
    assert out["sequences"][0] == [gc.BOS, flat_cell(4, 4), gc.EOS]
    assert out["sequences"][1] == [gc.BOS, 12, gc.EOS]
    assert out["driver_idxs"] == [0, 1]
