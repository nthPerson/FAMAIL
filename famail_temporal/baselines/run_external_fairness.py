"""Compute + report external fairness metrics before/after edit. See
docs/superpowers/plans/2026-07-02-external-fairness-metrics.md."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines import external_fairness_io as io

GROUPINGS: Tuple[str, str] = ("district_extremes", "median_split")


def _groups_for(values: np.ndarray, axis: str, grouping: str) -> np.ndarray:
    high = io.DISADVANTAGED_HIGH[axis]
    if grouping == "district_extremes":
        return ef.region_extremes(values, disadvantaged_high=high)
    if grouping == "median_split":
        return ef.median_split(values, disadvantaged_high=high)
    raise ValueError(f"unknown grouping {grouping!r}")


def assemble_results(
    Y_before: np.ndarray, Y_after: np.ndarray,
    demo: Dict[str, np.ndarray], seed: int = 0, B: int = 1000,
) -> dict:
    regions = ef.regions_from_values([demo[a] for a in io.EQUITY_AXES])
    specs: List[Tuple[str, object, np.ndarray]] = [("theil", ef.theil_index, regions)]
    metrics: Dict[str, dict] = {}
    for axis in io.EQUITY_AXES:
        metrics[axis] = {}
        for g in GROUPINGS:
            groups = _groups_for(demo[axis], axis, g)
            metrics[axis][g] = {
                "group_sizes": {
                    "n_disadvantaged": int((groups == 1).sum()),
                    "n_advantaged": int((groups == 0).sum()),
                    "n_excluded": int((groups == -1).sum()),
                },
                "supply_demand_ratio": {
                    "before": ef.supply_demand_ratio(Y_before, groups),
                    "after": ef.supply_demand_ratio(Y_after, groups),
                    "delta_gap": ef.sdr_gap(Y_after, groups) - ef.sdr_gap(Y_before, groups),
                },
                "demographic_parity": {
                    "before": ef.demographic_parity(Y_before, groups),
                    "after": ef.demographic_parity(Y_after, groups),
                    "delta": (ef.demographic_parity(Y_after, groups)
                              - ef.demographic_parity(Y_before, groups)),
                },
                "disparate_impact": {
                    "before": ef.disparate_impact(Y_before, groups),
                    "after": ef.disparate_impact(Y_after, groups),
                    "delta": (ef.disparate_impact(Y_after, groups)
                              - ef.disparate_impact(Y_before, groups)),
                },
            }
            specs.append((f"dp::{axis}::{g}", ef.demographic_parity, groups))
            specs.append((f"di::{axis}::{g}", ef.disparate_impact, groups))
            specs.append((f"sdrgap::{axis}::{g}", ef.sdr_gap, groups))

    boot = ef.paired_bootstrap(Y_before, Y_after, specs, B=B, seed=seed)

    # attach CIs
    theil_before = ef.theil_index(Y_before, regions)
    theil_after = ef.theil_index(Y_after, regions)
    result = {
        "theil": {
            "before": theil_before, "after": theil_after,
            "delta": theil_after - theil_before,
            "delta_ci": boot["theil"]["delta"],
            "n_dropped": boot["theil"]["n_dropped"],
        },
        "metrics": metrics,
    }
    for axis in io.EQUITY_AXES:
        for g in GROUPINGS:
            e = metrics[axis][g]
            e["demographic_parity"]["delta_ci"] = boot[f"dp::{axis}::{g}"]["delta"]
            e["disparate_impact"]["delta_ci"] = boot[f"di::{axis}::{g}"]["delta"]
            e["supply_demand_ratio"]["gap_ci"] = boot[f"sdrgap::{axis}::{g}"]["delta"]
    return result
