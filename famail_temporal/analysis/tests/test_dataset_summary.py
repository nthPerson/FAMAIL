from famail_temporal.analysis.dataset_summary import dataset_summary


def test_dataset_summary_pairs_dirty_and_clean():
    dirty = {"removal_summary": {"n_removed": 195840, "removal_rate": 0.4975,
                                  "total_seeking_extracted": 214286,
                                  "total_driving_extracted": 179384,
                                  "total_extracted": 393670}}
    clean = {"removal_summary": {"n_removed": 119290, "removal_rate": 0.3895,
                                  "total_seeking_extracted": 133091,
                                  "total_driving_extracted": 173178,
                                  "total_extracted": 306269},
             "stuck_gps_sinks": {"n_pickups_removed": 106677,
                                  "flagged_cells": [[17, 39], [29, 53]]}}
    s = dataset_summary(dirty, clean)
    assert s["dirty"]["removal_rate"] == 0.4975
    assert s["clean"]["removal_rate"] == 0.3895
    assert s["clean"]["n_sink_cells"] == 2
    assert s["clean"]["phantom_pickups_removed"] == 106677
    assert s["delta"]["removal_rate"] == round(0.3895 - 0.4975, 4)
    # The rate's true denominator (seeking + driving) is now surfaced.
    assert s["clean"]["total_extracted"] == 306269
    assert s["dirty"]["total_extracted"] == 393670
    assert s["clean"]["removal_rate_denominator"].startswith("total_extracted")


def test_total_extracted_derived_when_absent():
    """If total_extracted is missing, it is reconstructed from seeking+driving."""
    dirty = {"removal_summary": {"n_removed": 10, "removal_rate": 0.5,
                                  "total_seeking_extracted": 12,
                                  "total_driving_extracted": 8}}
    clean = {"removal_summary": {"n_removed": 5, "removal_rate": 0.25,
                                  "total_seeking_extracted": 11,
                                  "total_driving_extracted": 9},
             "stuck_gps_sinks": {"n_pickups_removed": 3, "flagged_cells": [[1, 2]]}}
    s = dataset_summary(dirty, clean)
    assert s["dirty"]["total_extracted"] == 20
    assert s["clean"]["total_extracted"] == 20
