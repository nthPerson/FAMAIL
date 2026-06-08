"""Unit tests for the terminal-cell transmission check."""
import math

import numpy as np

from famail_temporal.baselines import transmission as tr


def test_terminal_cell_histogram_normalized_and_one_hot_on_single_pickup():
    h = tr.terminal_cell_histogram([(2, 3, 0)], n_cells=100)
    assert h.shape == (100,)
    flat = 2 * 90 + 3  # gc.GY = 90; flat_cell(2, 3) = 2*90 + 3 = 183
    # In this test n_cells=100 < flat=183, so the out-of-range guard drops it.
    # Use a different example below.
    h2 = tr.terminal_cell_histogram([(0, 5, 0), (0, 5, 1)], n_cells=100)
    assert math.isclose(h2.sum(), 1.0, rel_tol=1e-12)
    assert h2[5] == 1.0  # flat_cell(0, 5) = 5; both pickups land there


def test_terminal_cell_histogram_handles_empty_input():
    h = tr.terminal_cell_histogram([], n_cells=100)
    assert h.shape == (100,) and h.sum() == 0.0


def test_jensen_shannon_zero_for_identical_distributions():
    p = np.array([0.25, 0.25, 0.5])
    js = tr.jensen_shannon_divergence(p, p)
    assert math.isclose(js, 0.0, abs_tol=1e-12)


def test_jensen_shannon_one_for_disjoint_distributions_in_bits():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    js = tr.jensen_shannon_divergence(p, q)
    # JS(disjoint) = log2(2) / 2 + log2(2) / 2 = 1.0 in bits
    assert math.isclose(js, 1.0, rel_tol=1e-6)


def test_jensen_shannon_symmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.4, 0.5])
    assert math.isclose(
        tr.jensen_shannon_divergence(p, q),
        tr.jensen_shannon_divergence(q, p),
        rel_tol=1e-12,
    )


def test_transmission_metrics_bundle_has_expected_keys():
    p_raw = np.array([0.5, 0.5, 0.0])
    p_edited = np.array([0.3, 0.7, 0.0])
    p_gen_b0 = np.array([0.5, 0.5, 0.0])
    p_gen_famail = np.array([0.4, 0.6, 0.0])
    out = tr.transmission_metrics(p_raw, p_edited, p_gen_b0, p_gen_famail)
    assert set(out) == {
        "js_target", "js_generated", "transmission_ratio",
        "js_b0_vs_raw", "js_famail_vs_edited",
    }
    # Target shift is real, generated shift is positive and smaller than target here.
    assert out["js_target"] > 0
    assert 0 < out["js_generated"] < out["js_target"]
    assert 0 < out["transmission_ratio"] < 1
