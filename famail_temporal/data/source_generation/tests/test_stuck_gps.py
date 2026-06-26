import pandas as pd
import pytest
from famail_temporal.data.source_generation import config as sgconfig
from famail_temporal.data.source_generation.stuck_gps import (
    pickup_mask, detect_stuck_gps_sinks, filter_stuck_gps_sinks, threshold_sensitivity,
)

def _df(rows):
    # rows: list of (plate_id, passenger_indicator, latitude, longitude, x_grid, y_grid)
    return pd.DataFrame(rows, columns=[
        "plate_id", "passenger_indicator", "latitude", "longitude", "x_grid", "y_grid",
    ])


def test_pickup_mask_flags_0_to_1_transitions_per_driver():
    df = _df([
        ("A", 0, 1.0, 1.0, 5, 5),
        ("A", 1, 1.0, 1.0, 5, 5),   # pickup (0->1)
        ("A", 0, 2.0, 2.0, 6, 6),   # dropoff (1->0), not a pickup
        ("B", 1, 9.0, 9.0, 1, 1),   # first row of B, diff is NaN -> not a pickup
    ])
    m = pickup_mask(df)
    assert list(m) == [False, True, False, False]


def test_detect_flags_a_concentrated_single_coord_sink():
    rows = []
    # driver SINK: 50 pickups frozen at one exact coord in cell (28,52)
    # The frozen coord has 0 dropoffs (all dropoffs are at indicator=0 rows which
    # are NOT the pickup coord). With max_dropoff_ratio=0.02, n_dropoffs=0 -> ratio=0.0 < 0.02 -> flagged.
    for _ in range(50):
        rows.append(("SINK", 0, 0.0, 0.0, 28, 52))
        rows.append(("SINK", 1, 12.345678, 98.765432, 28, 52))  # frozen pickup coord
    # SAME SINK driver: one legit pickup at a DIFFERENT coord/cell. This makes
    # the sink's per-coord count (50) differ from its driver-total (51), so the
    # assertions below prove driver_total = the driver's TOTAL pickups, not the
    # per-coord group size.
    rows.append(("SINK", 0, 0.0, 0.0, 7, 8))
    rows.append(("SINK", 1, 33.333333, 44.444444, 7, 8))
    # one normal pickup elsewhere in a different cell
    rows.append(("NORM", 0, 1.0, 1.0, 5, 5))
    rows.append(("NORM", 1, 1.111111, 2.222222, 5, 5))
    df = _df(rows)
    flagged, dist = detect_stuck_gps_sinks(
        df, min_pickups=10, max_dropoff_ratio=0.02, coord_precision=6,
    )
    assert len(flagged) == 1
    row = flagged.iloc[0]
    assert (int(row.x_grid), int(row.y_grid)) == (28, 52)
    assert int(row.n_pickups) == 50
    assert row.cell_share == 1.0
    # driver_total counts ALL of the driver's pickups (50 frozen + 1 legit),
    # not just the flagged coord group.
    assert int(row.driver_total) == 51
    # distribution is sorted desc; SINK has 2 coord groups + NORM has 1
    assert int(dist.iloc[0].n_pickups) == 50
    assert len(dist) == 3
    # new columns: n_dropoffs and dropoff_ratio present in both frames
    assert "n_dropoffs" in flagged.columns
    assert "dropoff_ratio" in flagged.columns
    assert "n_dropoffs" in dist.columns
    assert "dropoff_ratio" in dist.columns
    # frozen sink coord has 0 dropoffs -> ratio 0.0
    assert int(row.n_dropoffs) == 0
    assert float(row.dropoff_ratio) == 0.0


def test_detect_returns_empty_frames_when_no_pickups():
    # passenger_indicator never transitions 0->1, so there are no pickups.
    df = _df([
        ("A", 0, 1.0, 1.0, 5, 5),
        ("A", 0, 2.0, 2.0, 6, 6),
        ("B", 0, 9.0, 9.0, 1, 1),
    ])
    flagged, dist = detect_stuck_gps_sinks(
        df, min_pickups=10, max_dropoff_ratio=0.02, coord_precision=6,
    )
    assert len(flagged) == 0
    assert len(dist) == 0
    # frames are well-typed (expected columns present) even when empty.
    for col in ("n_pickups", "cell_share", "driver_total", "x_grid", "y_grid",
                "n_dropoffs", "dropoff_ratio"):
        assert col in flagged.columns
        assert col in dist.columns


def _sink_df():
    rows = []
    for _ in range(50):
        rows.append(("SINK", 0, 0.0, 0.0, 28, 52))
        rows.append(("SINK", 1, 12.345678, 98.765432, 28, 52))
    rows.append(("NORM", 0, 1.0, 1.0, 5, 5))
    rows.append(("NORM", 1, 1.111111, 2.222222, 5, 5))
    return _df(rows)


def test_filter_drops_flagged_pickups_and_keeps_drivers():
    df = _sink_df()
    cleaned, audit = filter_stuck_gps_sinks(
        df, min_pickups=10, max_dropoff_ratio=0.02, coord_precision=6, expected_cells=None,
    )
    # the 50 frozen pickup rows are gone; the SINK driver still exists (its 50 indicator-0 rows remain)
    assert cleaned["plate_id"].nunique() == 2
    assert audit["n_rows_removed"] == 50
    assert audit["flagged_cells"] == [(28, 52)]
    assert audit["sinks"][0]["n_pickups"] == 50
    # normal pickup survives
    assert ((cleaned["plate_id"] == "NORM") & (cleaned["passenger_indicator"] == 1)).sum() == 1
    # audit sinks carry n_dropoffs and dropoff_ratio
    assert "n_dropoffs" in audit["sinks"][0]
    assert "dropoff_ratio" in audit["sinks"][0]
    assert audit["sinks"][0]["n_dropoffs"] == 0
    assert audit["sinks"][0]["dropoff_ratio"] == 0.0


def test_hybrid_guard_asserts_expected_cells():
    df = _sink_df()
    # correct expectation passes
    filter_stuck_gps_sinks(df, min_pickups=10, max_dropoff_ratio=0.02, coord_precision=6,
                           expected_cells={(28, 52)})
    # wrong expectation raises
    with pytest.raises(AssertionError):
        filter_stuck_gps_sinks(df, min_pickups=10, max_dropoff_ratio=0.02, coord_precision=6,
                               expected_cells={(1, 1)})


def test_filter_is_noop_on_clean_data():
    df = _df([
        ("A", 0, 1.0, 1.0, 5, 5), ("A", 1, 1.1, 2.1, 5, 5),
        ("B", 0, 3.0, 3.0, 7, 7), ("B", 1, 3.1, 4.1, 7, 7),
    ])
    cleaned, audit = filter_stuck_gps_sinks(
        df, min_pickups=10, max_dropoff_ratio=0.02, coord_precision=6, expected_cells=set(),
    )
    assert len(cleaned) == len(df)
    assert audit["n_rows_removed"] == 0


def test_threshold_sensitivity_plateaus_then_drops():
    df = _sink_df()  # one 50-pickup sink + one 1-pickup normal cell
    curve = threshold_sensitivity(df, thresholds=[1, 10, 60], max_dropoff_ratio=0.02, coord_precision=6)
    by_t = {c["min_pickups"]: c["n_flagged_cells"] for c in curve}
    assert by_t[10] == 1     # only the sink
    assert by_t[60] == 0     # threshold above the sink size -> nothing
    assert by_t[1] >= 1
    # curve is monotone non-increasing in the threshold (the stability property)
    assert by_t[1] >= by_t[10] >= by_t[60]


def test_config_has_stuck_gps_constants():
    assert isinstance(sgconfig.STUCK_GPS_EXPECTED_CELLS, (set, frozenset))
    # dry-run-confirmed pipeline cells (10 distinct; +1 axis offset from the
    # documented famail-0-indexed sinks, e.g. documented (28,52) -> (29,53)).
    assert len(sgconfig.STUCK_GPS_EXPECTED_CELLS) == 10
    assert (29, 53) in sgconfig.STUCK_GPS_EXPECTED_CELLS
    assert sgconfig.STUCK_GPS_COORD_PRECISION == 6
    assert sgconfig.STUCK_GPS_DROP is True
    assert hasattr(sgconfig, "STUCK_GPS_MIN_PICKUPS")
    assert hasattr(sgconfig, "STUCK_GPS_MAX_DROPOFF_RATIO")
    assert not hasattr(sgconfig, "STUCK_GPS_COORD_DOMINANCE")


def test_real_hub_with_balanced_dropoffs_is_not_flagged():
    """A high-count coord with balanced dropoffs is NOT a stuck-GPS sink.

    The old cell-dominance rule could have wrongly caught real high-volume hubs.
    The new signature rule (near-zero dropoffs) correctly spares them.

    We build a "hub" driver with 200 pickups AND 200 dropoffs at the SAME exact
    coordinate. The sequence cycles:
        (0 @ offcoord) -> (1 @ HUB) -> (0 @ HUB) -> (1 @ offcoord) -> ...
    so HUB receives both 0->1 (pickup) and 1->0 (dropoff) transitions.
    With n_pickups=200 and n_dropoffs=200, dropoff_ratio=1.0 >= 0.02 -> NOT flagged.
    """
    HUB_LAT, HUB_LON = 12.345678, 98.765432
    OFF_LAT, OFF_LON = 0.0, 0.0
    rows = []
    # Cycle: off->HUB (pickup at HUB), HUB->off (dropoff at HUB)
    # Each pair = 1 pickup at HUB + 1 dropoff at HUB
    for _ in range(200):
        rows.append(("HUB", 0, OFF_LAT, OFF_LON, 5, 5))   # seeking at off-coord
        rows.append(("HUB", 1, HUB_LAT, HUB_LON, 28, 52)) # pickup at HUB (0->1)
        rows.append(("HUB", 0, HUB_LAT, HUB_LON, 28, 52)) # dropoff at HUB (1->0)
        rows.append(("HUB", 1, OFF_LAT, OFF_LON, 5, 5))   # pickup at off (0->1, NOT HUB)
    df = _df(rows)

    flagged, dist = detect_stuck_gps_sinks(
        df, min_pickups=10, max_dropoff_ratio=0.02, coord_precision=6,
    )
    # HUB coord has 200 pickups but also 200 dropoffs -> ratio=1.0 -> NOT flagged
    assert len(flagged) == 0, (
        f"Real hub with balanced dropoffs was wrongly flagged: {flagged.to_dict(orient='records')}"
    )
    # Verify the hub group IS in the distribution (with n_dropoffs > 0)
    hub_rows = dist[
        (dist["lat_r"] == round(HUB_LAT, 6)) & (dist["lon_r"] == round(HUB_LON, 6))
    ]
    assert len(hub_rows) == 1
    assert int(hub_rows.iloc[0]["n_pickups"]) == 200
    assert int(hub_rows.iloc[0]["n_dropoffs"]) == 200
    assert float(hub_rows.iloc[0]["dropoff_ratio"]) == pytest.approx(1.0)


from famail_temporal.data.source_generation import cli as sgcli

def test_report_stuck_gps_returns_audit_and_curve(monkeypatch):
    df = _sink_df()
    # stub the heavy load step so the dry-run runs on synthetic data
    monkeypatch.setattr(sgcli, "_load_event_df_for_report", lambda _in: df)
    # inject min_pickups=10 so the 50-pickup synthetic sink flags but the
    # 1-pickup NORM cell does not (production default is config.STUCK_GPS_MIN_PICKUPS).
    rep = sgcli.report_stuck_gps("ignored", expected_cells=None, min_pickups=10)
    assert rep["audit"]["flagged_cells"] == [(28, 52)]
    assert rep["distribution_top"][0]["n_pickups"] == 50
    assert any(c["min_pickups"] for c in rep["threshold_curve"])
