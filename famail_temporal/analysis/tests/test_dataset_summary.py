from famail_temporal.analysis.dataset_summary import dataset_summary


def test_dataset_summary_pairs_dirty_and_clean():
    dirty = {"removal_summary": {"n_removed": 195840, "removal_rate": 0.4975,
                                  "total_seeking_extracted": 214286}}
    clean = {"removal_summary": {"n_removed": 119290, "removal_rate": 0.3895,
                                  "total_seeking_extracted": 133091},
             "stuck_gps_sinks": {"n_pickups_removed": 106677,
                                  "flagged_cells": [[17, 39], [29, 53]]}}
    s = dataset_summary(dirty, clean)
    assert s["dirty"]["removal_rate"] == 0.4975
    assert s["clean"]["removal_rate"] == 0.3895
    assert s["clean"]["n_sink_cells"] == 2
    assert s["clean"]["phantom_pickups_removed"] == 106677
    assert s["delta"]["removal_rate"] == round(0.3895 - 0.4975, 4)
