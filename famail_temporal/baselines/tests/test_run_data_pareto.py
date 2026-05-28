"""Unit test for the run_data_pareto edited-point adapter."""
from types import SimpleNamespace

from famail_temporal.baselines import run_data_pareto as rdp


def test_edited_point_from_result_reads_after_fields():
    fake = SimpleNamespace(
        f_spatial_after=0.083, f_causal_after=0.814,
        gini_dsr_after=0.91, gini_asr_after=0.90,
    )
    pt = rdp.edited_point_from_result(fake)
    assert pt.label == "edit"
    assert pt.retention == 1.0
    assert pt.f_causal == 0.814
    assert pt.f_spatial == 0.083
