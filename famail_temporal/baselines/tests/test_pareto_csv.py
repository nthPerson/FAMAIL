"""TDD test for pareto CSV mirror and removed-ids (E15/E17)."""
from dataclasses import asdict
from famail_temporal.baselines.pareto import ParetoPoint, points_to_csv_rows


def test_points_to_csv_rows_flat():
    pts = [ParetoPoint("raw", 1.0, 0.10, 0.81, 0.7, 0.8, 0),
           ParetoPoint("filter@100", 0.999, 0.11, 0.82, 0.69, 0.8, 100)]
    rows = points_to_csv_rows(pts)
    assert rows[0]["label"] == "raw" and rows[1]["n_removed"] == 100
    assert set(rows[0]) == set(asdict(pts[0]))
