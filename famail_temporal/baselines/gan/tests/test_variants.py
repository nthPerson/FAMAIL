"""Unit tests for gan.variants."""
import pickle
from types import SimpleNamespace

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.datasets import rank_unfair_trajectory_indices
from famail_temporal.baselines.gan import variants


def test_apply_edits_swaps_by_trajectory_id_preserving_order():
    raw = [make_traj_at(1, 1, 0, traj_id=10),
           make_traj_at(2, 2, 0, traj_id=11),
           make_traj_at(3, 3, 0, traj_id=12)]
    edited_11 = make_traj_at(5, 5, 0, traj_id=11)
    out = variants.apply_edits(raw, {11: edited_11})
    assert [t.trajectory_id for t in out] == [10, 11, 12]
    assert out[1] is edited_11
    assert out[0] is raw[0] and out[2] is raw[2]


def test_load_edited_trajectories_reads_histories_pkl(tmp_path):
    bundle = _make_synthetic_bundle()
    bundle.trajectories.extend([
        make_traj_at(2, 2, 0, traj_id=100),
        make_traj_at(3, 3, 0, traj_id=101),
    ])
    edited_100 = make_traj_at(4, 4, 0, traj_id=100)
    histories = [SimpleNamespace(modified=edited_100)]
    (tmp_path / "histories.pkl").write_bytes(pickle.dumps(histories))

    out = variants.load_edited_trajectories(bundle, tmp_path)
    assert len(out) == len(bundle.trajectories)
    by_id = {t.trajectory_id: t for t in out}
    # `==` not `is`: histories.pkl round-trips through pickle, so the loaded
    # `.modified` is a structurally-equal copy, never the literal pre-pickle
    # object. Trajectory is a plain dataclass → field-wise equality.
    assert by_id[100] == edited_100
    assert by_id[101] in bundle.trajectories


def test_filtered_trajectories_drops_top_ranked():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    ranked = rank_unfair_trajectory_indices(bundle)
    n = min(2, len(ranked))
    out = variants.filtered_trajectories(bundle, n)
    assert len(out) == len(bundle.trajectories) - n
    removed_ids = {bundle.trajectories[i].trajectory_id for i in ranked[:n]}
    kept_ids = {t.trajectory_id for t in out}
    assert kept_ids.isdisjoint(removed_ids)


def test_filtered_trajectories_zero_remove_is_full_corpus():
    bundle = _make_synthetic_bundle()
    out = variants.filtered_trajectories(bundle, 0)
    assert len(out) == len(bundle.trajectories)
