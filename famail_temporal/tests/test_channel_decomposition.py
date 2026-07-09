"""Unit tests for the supply-vs-demand channel decomposition (Task 11a review).

Uses a tiny synthetic grid (SimpleNamespace bundle double exposing exactly the
two attributes ``service_ratio_Y`` reads: ``mask_3d`` + ``active_taxis_3d``)
with hand-set S / S' / S_tier2 / D / D' grids, so every channel value is known
by construction.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.baselines import external_fairness_io as efio
from famail_temporal.analysis.channel_decomposition import (
    bootstrap_channels,
    compute_channel_vectors,
)


# --- tiny synthetic fixture ---------------------------------------------------
# 2x2x1 grid, all cells active. Unit order (C order over mask):
#   u0=(0,0) u1=(0,1) u2=(1,0) u3=(1,1). D-group = {u0, u1}.

GX, GY, T = 2, 2, 1


def _grid(vals):
    """vals: 4 numbers in unit order -> (2,2,1) float grid."""
    return np.asarray(vals, dtype=np.float64).reshape(GX, GY, T)


def _bundle_double(S_grid):
    mask = np.ones((GX, GY, T), dtype=bool)
    return SimpleNamespace(mask_3d=mask, active_taxis_3d=S_grid)


def _Y(D_grid, S_grid):
    """Per-unit Y = S / max(D, DEMAND_FLOOR) via the real service_ratio_Y."""
    return efio.service_ratio_Y(D_grid, _bundle_double(S_grid))


@pytest.fixture()
def grids():
    # Demand well above DEMAND_FLOOR (0.5) everywhere so Y responds linearly.
    S_base = _grid([2.0, 4.0, 6.0, 8.0])
    D_base = _grid([2.0, 2.0, 4.0, 4.0])
    # Edit: raise supply in u0 (+1), lower in u3 (-2); move demand out of u1
    # (2.0 -> 1.0, still above floor) and into u2 (4.0 -> 5.0).
    S_prime = _grid([3.0, 4.0, 6.0, 6.0])
    D_prime = _grid([2.0, 1.0, 5.0, 4.0])
    # Distinct-count after-supply: a different S' (u0 +0.5 only).
    S_tier2 = _grid([2.5, 4.0, 6.0, 8.0])
    d_mask = np.array([True, True, False, False])  # D-group = {u0, u1}
    return S_base, S_prime, S_tier2, D_base, D_prime, d_mask


def _vectors(grids, with_tier2=False):
    S_base, S_prime, S_tier2, D_base, D_prime, d = grids
    Y_bb = _Y(D_base, S_base)[d]
    Y_bp = _Y(D_prime, S_base)[d]
    Y_pp = _Y(D_prime, S_prime)[d]
    Y_pb = _Y(D_base, S_prime)[d]
    Y_t2 = _Y(D_prime, S_tier2)[d] if with_tier2 else None
    return Y_bb, Y_bp, Y_pp, Y_pb, Y_t2


# --- (a) telescoping identities -------------------------------------------------

def test_telescoping_demand_first(grids):
    Y_bb, Y_bp, Y_pp, Y_pb, _ = _vectors(grids)
    ch = compute_channel_vectors(Y_bb, Y_bp, Y_pp, Y_pb)
    np.testing.assert_allclose(ch["demand"] + ch["supply"], ch["total"], rtol=0, atol=0)


def test_telescoping_supply_first(grids):
    Y_bb, Y_bp, Y_pp, Y_pb, _ = _vectors(grids)
    ch = compute_channel_vectors(Y_bb, Y_bp, Y_pp, Y_pb)
    np.testing.assert_allclose(
        ch["supply_first"] + ch["demand_second"], ch["total"], rtol=0, atol=0,
    )


# --- (b) tier-2 substitution ------------------------------------------------------

def test_tier2_substitutes_supply_grid_only(grids):
    Y_bb, Y_bp, Y_pp, Y_pb, Y_t2 = _vectors(grids, with_tier2=True)
    ch = compute_channel_vectors(Y_bb, Y_bp, Y_pp, Y_pb, Y_t2=Y_t2)
    ch_no_t2 = compute_channel_vectors(Y_bb, Y_bp, Y_pp, Y_pb)
    # tier-2 supply channel = Y(S_tier2, D') - Y(S_base, D'), by construction.
    np.testing.assert_allclose(ch["supply_tier2"], Y_t2 - Y_bp)
    np.testing.assert_allclose(ch["total_tier2"], Y_t2 - Y_bb)
    # Demand channel is UNCHANGED by tier-2 mode.
    np.testing.assert_allclose(ch["demand"], ch_no_t2["demand"], rtol=0, atol=0)
    # And tier-2 keys are absent without a tier-2 grid.
    assert "supply_tier2" not in ch_no_t2 and "total_tier2" not in ch_no_t2


def test_tier2_telescoping(grids):
    Y_bb, Y_bp, Y_pp, Y_pb, Y_t2 = _vectors(grids, with_tier2=True)
    ch = compute_channel_vectors(Y_bb, Y_bp, Y_pp, Y_pb, Y_t2=Y_t2)
    np.testing.assert_allclose(
        ch["demand"] + ch["supply_tier2"], ch["total_tier2"], rtol=0, atol=0,
    )


# --- (c) sign conventions ---------------------------------------------------------

def test_raising_supply_in_group_cell_is_positive_supply_channel(grids):
    S_base, _, _, D_base, D_prime, d = grids
    # Supply-only edit: +1 taxi in u0 (a D-group cell), demand untouched.
    S_up = _grid([3.0, 4.0, 6.0, 8.0])
    Y_bb = _Y(D_base, S_base)[d]
    Y_bp = _Y(D_base, S_base)[d]          # D' == D_base (no demand move)
    Y_pp = _Y(D_base, S_up)[d]
    Y_pb = _Y(D_base, S_up)[d]
    ch = compute_channel_vectors(Y_bb, Y_bp, Y_pp, Y_pb)
    assert ch["supply"].mean() > 0
    assert ch["demand"].mean() == 0.0
    # u0: dY = +1/2 = 0.5 exactly; u1 unchanged -> group mean 0.25.
    np.testing.assert_allclose(ch["supply"].mean(), 0.25)


def test_moving_demand_out_of_group_cell_is_positive_demand_channel(grids):
    S_base, _, _, D_base, _, d = grids
    # Demand-only edit: move 1.0 of demand OUT of u1 (D-group, 2.0 -> 1.0,
    # stays above DEMAND_FLOOR=0.5) into u2 (non-group). Supply untouched.
    assert config.DEMAND_FLOOR == 0.5
    D_out = _grid([2.0, 1.0, 5.0, 4.0])
    Y_bb = _Y(D_base, S_base)[d]
    Y_bp = _Y(D_out, S_base)[d]
    Y_pp = _Y(D_out, S_base)[d]           # S' == S_base (no supply move)
    Y_pb = _Y(D_base, S_base)[d]
    ch = compute_channel_vectors(Y_bb, Y_bp, Y_pp, Y_pb)
    assert ch["demand"].mean() > 0
    assert ch["supply"].mean() == 0.0
    # u1: Y goes 4/2=2 -> 4/1=4, dY=+2; u0 unchanged -> group mean +1.
    np.testing.assert_allclose(ch["demand"].mean(), 1.0)


# --- (d) bootstrap: shared index draw per replicate --------------------------------

def test_bootstrap_identical_grids_gives_exact_zero_cis(grids):
    S_base, _, _, D_base, _, d = grids
    Y = _Y(D_base, S_base)[d]
    ch = compute_channel_vectors(Y, Y, Y, Y)          # no edit at all
    boot = bootstrap_channels(ch, B=200, seed=0)
    for name, c in boot.items():
        assert c["point"] == 0.0, name
        assert (c["ci_lo"], c["ci_hi"]) == (0.0, 0.0), name
        assert c["significant"] is False, name


def test_bootstrap_shares_one_index_draw_across_channels():
    """With B = -A per unit, a SHARED resample per replicate forces every
    replicate mean of B to equal -mean of A exactly, so the CIs must mirror:
    ci_lo(B) == -ci_hi(A) and ci_hi(B) == -ci_lo(A) (up to percentile
    interpolation rounding, ~1e-17). Independent draws per channel would
    differ at the CI-width scale (~1e-2), decades above the tolerance."""
    rng = np.random.default_rng(42)
    a = rng.normal(size=97)
    boot = bootstrap_channels({"A": a, "B": -a}, B=500, seed=7)
    np.testing.assert_allclose(boot["B"]["ci_lo"], -boot["A"]["ci_hi"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(boot["B"]["ci_hi"], -boot["A"]["ci_lo"], rtol=0, atol=1e-12)


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bootstrap_channels({"A": np.zeros(5), "B": np.zeros(4)}, B=10)
