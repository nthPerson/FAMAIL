"""TDD test for per-sink F_spatial decomposition (E23)."""
import numpy as np
from famail_temporal.analysis.sink_decomposition import sink_spatial_contributions


def test_sink_contribution_sums_active_spatial_channel():
    dense = np.zeros((4, 4, 2), dtype=np.float32)   # (gx,gy,T) spatial channel
    mask = np.zeros((4, 4, 2), dtype=bool)
    dense[2, 3, 0] = 0.05; mask[2, 3, 0] = True
    dense[2, 3, 1] = 0.02; mask[2, 3, 1] = True
    dense[2, 3, 1] = 0.02
    # sink at 1-indexed (3,4) -> 0-indexed (2,3); sum over active t = 0.07
    out = sink_spatial_contributions(dense, mask, [(3, 4)])
    assert round(out["per_sink"]["(3, 4)"], 4) == 0.07
    assert round(out["total"], 4) == 0.07


def test_sink_contribution_ignores_inactive_t_blocks():
    dense = np.zeros((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4, 3), dtype=bool)
    # Only t=0 is active for this sink; t=1 has data but mask=False
    dense[0, 0, 0] = 0.10; mask[0, 0, 0] = True
    dense[0, 0, 1] = 0.99  # inactive; should NOT be summed
    out = sink_spatial_contributions(dense, mask, [(1, 1)])
    assert round(out["per_sink"]["(1, 1)"], 4) == 0.10


def test_multiple_sinks_total_is_sum_of_per_sink():
    dense = np.zeros((5, 5, 2), dtype=np.float32)
    mask = np.zeros((5, 5, 2), dtype=bool)
    dense[0, 0, 0] = 0.03; mask[0, 0, 0] = True
    dense[1, 1, 0] = 0.07; mask[1, 1, 0] = True
    out = sink_spatial_contributions(dense, mask, [(1, 1), (2, 2)])
    per = out["per_sink"]
    assert abs(out["total"] - sum(per.values())) < 1e-6


def test_sink_with_no_active_t_returns_zero():
    dense = np.zeros((4, 4, 2), dtype=np.float32)
    mask = np.zeros((4, 4, 2), dtype=bool)
    dense[2, 2, 0] = 0.5  # data but mask=False everywhere
    out = sink_spatial_contributions(dense, mask, [(3, 3)])
    assert out["per_sink"]["(3, 3)"] == 0.0
    assert out["total"] == 0.0
