"""Smoke test for the Pareto figure."""
from famail_temporal.baselines.pareto import ParetoPoint
from famail_temporal.baselines import figure as fig


def test_plot_pareto_writes_png(tmp_path):
    points = [
        ParetoPoint("raw", 1.0, 0.08, 0.805, 0.92, 0.91, 0),
        ParetoPoint("filter@100", 0.99, 0.08, 0.808, 0.92, 0.91, 100),
        ParetoPoint("filter@500", 0.95, 0.08, 0.815, 0.92, 0.91, 500),
        ParetoPoint("edit", 1.0, 0.08, 0.814, 0.92, 0.91, 0),
    ]
    out = tmp_path / "pareto.png"
    fig.plot_pareto(points, out, metric="f_causal")
    assert out.exists() and out.stat().st_size > 0
