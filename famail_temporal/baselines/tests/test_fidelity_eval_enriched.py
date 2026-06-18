import numpy as np

from famail_temporal.baselines import fidelity_eval as fe


def test_trajectory_statistics_has_new_keys():
    # straight line (0,0)->(3,0): RoG of x-coords {0,1,2,3} about mean 1.5,
    # y all 0 -> RoG = sqrt(mean((x-1.5)^2)) = sqrt(1.25); net disp = 3.0
    cells = [fe.gc.GY * 0 + 0, fe.gc.GY * 1 + 0, fe.gc.GY * 2 + 0, fe.gc.GY * 3 + 0]
    s = fe.trajectory_statistics(cells)
    assert set(s) >= {"length", "mean_displacement", "coverage",
                      "radius_of_gyration", "net_displacement"}
    assert abs(s["net_displacement"] - 3.0) < 1e-6
    assert abs(s["radius_of_gyration"] - np.sqrt(1.25)) < 1e-6


def test_short_trajectory_zero_rog_and_netdisp():
    s = fe.trajectory_statistics([fe.gc.GY * 5 + 5])   # length 1
    assert s["radius_of_gyration"] == 0.0
    assert s["net_displacement"] == 0.0


def test_distributional_fidelity_default_keys_unchanged():
    """Default (3-key) aggregate is unchanged (v1 backward-compat)."""
    src = [{"length": 2, "mean_displacement": 1.0, "coverage": 2,
            "radius_of_gyration": 0.5, "net_displacement": 1.0}]
    raw = [{"length": 2, "mean_displacement": 1.0, "coverage": 2,
            "radius_of_gyration": 0.5, "net_displacement": 1.0}]
    out = fe.distributional_fidelity(src, raw)
    assert set(out["per_stat"]) == set(fe._STAT_KEYS)   # only the original 3
    assert out["aggregate"] == 0.0                       # identical -> 0


def test_distributional_fidelity_v2_keys():
    src = [{"length": 2, "mean_displacement": 1.0, "coverage": 2,
            "radius_of_gyration": 0.5, "net_displacement": 1.0}]
    raw = [{"length": 9, "mean_displacement": 4.0, "coverage": 9,
            "radius_of_gyration": 3.0, "net_displacement": 8.0}]
    out = fe.distributional_fidelity(src, raw, keys=fe._STAT_KEYS_V2)
    assert set(out["per_stat"]) == set(fe._STAT_KEYS_V2)  # all 5
    assert out["aggregate"] > 0.0


def test_terminal_cell_distribution_js_identical_zero():
    pk = [(1, 2, 0), (3, 4, 1), (1, 2, 2)]
    assert fe.terminal_cell_distribution_js(pk, pk) == 0.0


def test_terminal_cell_distribution_js_disjoint_high():
    a = [(1, 2, 0), (1, 2, 1)]
    b = [(10, 20, 0), (10, 20, 1)]
    js = fe.terminal_cell_distribution_js(a, b)
    assert js > 0.9   # disjoint support -> ~1 bit
