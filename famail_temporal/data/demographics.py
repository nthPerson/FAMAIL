"""
Derive per-capita and log-transformed demographic features from raw columns.

The raw cell_demographics.pkl contains 13 district-level economic/demographic
columns. Several configured features (GDPperCapita, CompPerCapita) are ratios
derived from these raw columns, not stored directly. This module computes
those derived features so that config.DEMOGRAPHIC_FEATURES can reference them.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np


def enrich_demographics(
    demo_grid: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Derive per-capita and log features from the raw demographics grid.

    Appends derived columns to the grid and returns the enriched grid with
    an expanded feature-name list. Cells with zero population get NaN for
    per-capita features (caught downstream by the active-mask NaN filter).

    Derived features:
      - GDPperCapita: GDPin10000Yuan / (YearEndPermanentPop10k * 10000)
      - CompPerCapita: EmployeeCompensation100MYuan * 1e8 / AvgEmployedPersons
      - MigrantRatio: NonRegisteredPermanentPop10k / YearEndPermanentPop10k
      - LogGDP: log1p(GDPin10000Yuan)
      - LogHousingPrice: log1p(AvgHousingPricePerSqM)
      - LogCompensation: log1p(EmployeeCompensation100MYuan)
      - LogPopDensity: log1p(PopDensityPerKm2)

    Args:
        demo_grid: (grid_x, grid_y, n_raw_features) float array, NaN for
            unmapped cells
        feature_names: raw feature names matching the last axis of demo_grid

    Returns:
        (enriched_grid, enriched_names) with derived features appended.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    derived = []
    derived_names = []
    eps = 1e-10

    # GDP per capita (Yuan per person)
    if "GDPin10000Yuan" in name_to_idx and "YearEndPermanentPop10k" in name_to_idx:
        gdp = demo_grid[:, :, name_to_idx["GDPin10000Yuan"]]
        pop = demo_grid[:, :, name_to_idx["YearEndPermanentPop10k"]]
        gdp_pc = gdp / (pop * 10000 + eps)
        derived.append(gdp_pc)
        derived_names.append("GDPperCapita")

    # Compensation per capita (Yuan per employed person)
    if "EmployeeCompensation100MYuan" in name_to_idx and "AvgEmployedPersons" in name_to_idx:
        comp = demo_grid[:, :, name_to_idx["EmployeeCompensation100MYuan"]]
        emp = demo_grid[:, :, name_to_idx["AvgEmployedPersons"]]
        comp_pc = comp * 1e8 / (emp + eps)
        derived.append(comp_pc)
        derived_names.append("CompPerCapita")

    # Migrant ratio
    if "NonRegisteredPermanentPop10k" in name_to_idx and "YearEndPermanentPop10k" in name_to_idx:
        non_reg = demo_grid[:, :, name_to_idx["NonRegisteredPermanentPop10k"]]
        total = demo_grid[:, :, name_to_idx["YearEndPermanentPop10k"]]
        migrant = non_reg / (total + eps)
        derived.append(migrant)
        derived_names.append("MigrantRatio")

    # Log transforms
    log_features = {
        "GDPin10000Yuan": "LogGDP",
        "AvgHousingPricePerSqM": "LogHousingPrice",
        "EmployeeCompensation100MYuan": "LogCompensation",
        "PopDensityPerKm2": "LogPopDensity",
    }
    for raw_name, log_name in log_features.items():
        if raw_name in name_to_idx:
            raw_vals = demo_grid[:, :, name_to_idx[raw_name]]
            derived.append(np.log1p(np.maximum(raw_vals, 0)))
            derived_names.append(log_name)

    if not derived:
        return demo_grid.copy(), list(feature_names)

    derived_stack = np.stack(derived, axis=-1)
    enriched_grid = np.concatenate([demo_grid, derived_stack], axis=-1)
    enriched_names = list(feature_names) + derived_names

    return enriched_grid, enriched_names
