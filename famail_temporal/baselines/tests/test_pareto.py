"""Unit tests for famail_temporal.baselines.pareto."""
import json

import pytest

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import (
    active_units, make_traj_at, negative_attribution_units,
)
from famail_temporal.baselines import pareto as p


def _bundle_with_trajs():
    bundle = _make_synthetic_bundle()
    units = negative_attribution_units(bundle, 5) + active_units(bundle, 10)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    return bundle


def test_raw_point_has_full_retention():
    bundle = _bundle_with_trajs()
    pt = p.raw_point(bundle)
    assert pt.label == "raw"
    assert pt.retention == 1.0
    assert pt.n_removed == 0


def test_filtered_points_retention_math():
    bundle = _bundle_with_trajs()
    n = len(bundle.trajectories)
    pts = p.filtered_points(bundle, k_levels=[1, 3])
    assert [pt.label for pt in pts] == ["filter@1", "filter@3"]
    assert pts[0].retention == pytest.approx((n - pts[0].n_removed) / n)
    # More filtering => lower (or equal) retention.
    assert pts[1].retention <= pts[0].retention


def test_filtered_k_capped_at_candidate_count():
    bundle = _bundle_with_trajs()
    huge = 10 ** 9
    pts = p.filtered_points(bundle, k_levels=[huge])
    assert pts[0].n_removed <= len(bundle.trajectories)


def test_edited_point_is_full_retention():
    pt = p.edited_point(
        f_spatial=0.10, f_causal=0.81, gini_dsr=0.9, gini_asr=0.9,
    )
    assert pt.label == "edit"
    assert pt.retention == 1.0
    assert pt.n_removed == 0


def test_points_to_json_roundtrips():
    bundle = _bundle_with_trajs()
    pts = [p.raw_point(bundle)] + p.filtered_points(bundle, [1])
    blob = p.points_to_json(pts)
    loaded = json.loads(blob)
    assert isinstance(loaded, list)
    assert loaded[0]["label"] == "raw"
    assert "f_causal" in loaded[0]
