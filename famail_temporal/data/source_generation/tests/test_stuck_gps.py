import pandas as pd
from famail_temporal.data.source_generation.stuck_gps import (
    pickup_mask, detect_stuck_gps_sinks,
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
    for _ in range(50):
        rows.append(("SINK", 0, 0.0, 0.0, 28, 52))
        rows.append(("SINK", 1, 12.345678, 98.765432, 28, 52))  # frozen pickup coord
    # one normal pickup elsewhere in a different cell
    rows.append(("NORM", 0, 1.0, 1.0, 5, 5))
    rows.append(("NORM", 1, 1.111111, 2.222222, 5, 5))
    df = _df(rows)
    flagged, dist = detect_stuck_gps_sinks(
        df, min_pickups=10, coord_dominance=0.9, coord_precision=6,
    )
    assert len(flagged) == 1
    row = flagged.iloc[0]
    assert (int(row.x_grid), int(row.y_grid)) == (28, 52)
    assert int(row.n_pickups) == 50
    assert row.cell_share == 1.0
    # distribution is sorted desc and includes both groups
    assert int(dist.iloc[0].n_pickups) == 50
    assert len(dist) == 2
