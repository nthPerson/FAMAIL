from famail_temporal.analysis.cleanup_delta import editor_delta


def test_editor_delta_isolates_baseline_shift_and_edit_robustness():
    dirty = {"metrics_before": {"f_spatial": 0.0822, "f_causal": 0.8052},
             "metrics_after":  {"f_spatial": 0.0825, "f_causal": 0.8180},
             "deltas": {"f_spatial": 0.0003, "f_causal": 0.0128}}
    clean = {"metrics_before": {"f_spatial": 0.1034, "f_causal": 0.8069},
             "metrics_after":  {"f_spatial": 0.1025, "f_causal": 0.8193},
             "deltas": {"f_spatial": -0.0009, "f_causal": 0.0124}}
    d = editor_delta(dirty, clean)
    # F_spatial baseline rose by the sink removal (~+0.021)
    assert round(d["f_spatial"]["baseline_shift_dirty_to_clean"], 4) == round(0.1034 - 0.0822, 4)
    # F_causal edit delta is ~unchanged (robust): |Δ_clean - Δ_dirty| small
    assert abs(d["f_causal"]["edit_delta_shift"]) < 0.001
