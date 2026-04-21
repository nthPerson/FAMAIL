"""
Preprocess source_data/ → cache/.

Run:    python -m famail_temporal.preprocess
Force:  python -m famail_temporal.preprocess --force

Pipeline phases (one-time cost):
  1. Load 4 raw .pkl files (pickup/dropoff counts, active taxis, demographics,
     district mapping).
  2. Aggregate raw dicts into (48, 90, T) mean-hourly tensors.
  3. Compute the active-unit mask (supply threshold + valid cells + finite
     demographics) and verify per-block/total minimums.
  4. Build the canonical UnitIndexMap ordering.
  5. Extract per-unit demand (D), supply (S), and demographic vectors.
  6. Fit g_0(D) on (D, Y = S/D) using the power basis.
  7. Precompute hat matrices (I - H_demo, M) for pooled F_causal.

All artifacts are written through data.cache_io with config-encoded filenames
so multiple configurations coexist peacefully.
"""

from __future__ import annotations
import argparse
import sys

import numpy as np

from famail_temporal import config
from famail_temporal.data.active_mask import UnitIndexMap, compute_active_mask
from famail_temporal.data.aggregation import (
    aggregate_pickup_dropoff,
    aggregate_active_taxis,
    dataset_n_days,
)
from famail_temporal.data.cache_io import cache_path, load_raw, save_artifact
from famail_temporal.data.demographics import enrich_demographics
from famail_temporal.fairness.g0_power_basis import fit as fit_g0
from famail_temporal.fairness.hat_matrices import precompute_hat_matrices
from famail_temporal.utils.seeding import set_all_seeds


def run(force: bool = False) -> None:
    """Execute the full preprocessing pipeline, writing artifacts to cache/."""
    set_all_seeds(config.DEFAULT_SEED)

    def _should_write(name: str, include_features: bool = False) -> bool:
        if force:
            return True
        return not cache_path(name, include_features).exists()

    if not force:
        print(
            "[preprocess] NOTE: running without --force. Existing cache "
            "artifacts will be kept even if raw data has changed. Use "
            "--force to regenerate everything.",
            flush=True,
        )

    # ---------------------------------------------------------------------
    # Phase 1: Load raw data
    # ---------------------------------------------------------------------
    print("[preprocess] Loading raw data ...", flush=True)
    pickup_dropoff_raw = load_raw("pickup_dropoff_counts.pkl")
    active_taxis_raw = load_raw("active_taxis_5x5_hourly.pkl")
    demographics_raw = load_raw("cell_demographics.pkl")
    district_raw = load_raw("grid_to_district_mapping.pkl")

    demographics_grid = demographics_raw['demographics_grid']
    demo_feature_names = list(demographics_raw['feature_names'])
    valid_mask = district_raw['valid_mask']

    # Derive per-capita and log features from raw demographic columns.
    # config.DEMOGRAPHIC_FEATURES may reference derived features (e.g.,
    # GDPperCapita, CompPerCapita) that don't exist in the raw data.
    demographics_grid, demo_feature_names = enrich_demographics(
        demographics_grid, demo_feature_names,
    )
    print(
        f"[preprocess] Demographics enriched: {len(demo_feature_names)} features "
        f"(raw + derived)",
        flush=True,
    )

    for feat in config.DEMOGRAPHIC_FEATURES:
        if feat not in demo_feature_names:
            raise ValueError(
                f"Demographic feature '{feat}' not found in enriched demographics: "
                f"{demo_feature_names}"
            )

    # active_taxis_5x5_hourly.pkl is sometimes saved as a dict bundle with a
    # 'data' key wrapping the actual counts dict — unwrap if so.
    if isinstance(active_taxis_raw, dict) and 'data' in active_taxis_raw:
        active_taxis_raw = active_taxis_raw['data']

    n_days = max(
        dataset_n_days(pickup_dropoff_raw),
        dataset_n_days(active_taxis_raw),
    )
    print(f"[preprocess] n_days = {n_days}", flush=True)

    if _should_write("metadata"):
        save_artifact("metadata", {
            'n_days': n_days,
            'config_T': config.T,
            'config_GRID_DIMS': config.GRID_DIMS,
            'config_ACTIVE_SUPPLY_THRESHOLD': config.ACTIVE_SUPPLY_THRESHOLD,
            'config_DEMAND_FLOOR': config.DEMAND_FLOOR,
            'config_DEMOGRAPHIC_FEATURES': list(config.DEMOGRAPHIC_FEATURES),
        })

    # ---------------------------------------------------------------------
    # Phase 2: Aggregate to (48, 90, T)
    # ---------------------------------------------------------------------
    print("[preprocess] Aggregating to (48, 90, T) ...", flush=True)
    pickup_3d, dropoff_3d = aggregate_pickup_dropoff(
        pickup_dropoff_raw, n_days=n_days,
    )
    active_taxis_3d = aggregate_active_taxis(active_taxis_raw, n_days=n_days)

    if _should_write("pickup_counts"):
        save_artifact("pickup_counts", pickup_3d)
    if _should_write("dropoff_counts"):
        save_artifact("dropoff_counts", dropoff_3d)
    if _should_write("active_taxis"):
        save_artifact("active_taxis", active_taxis_3d)

    # ---------------------------------------------------------------------
    # Phase 3: Active-unit mask + minimum guards
    # ---------------------------------------------------------------------
    print("[preprocess] Computing active mask ...", flush=True)
    feat_indices = [demo_feature_names.index(f)
                    for f in config.DEMOGRAPHIC_FEATURES]
    demographics_selected = demographics_grid[..., feat_indices].astype(np.float32)
    mask_3d = compute_active_mask(
        active_taxis_3d, valid_mask, demographics_selected,
    )

    n_total = int(mask_3d.sum())
    assert n_total >= config.MIN_TOTAL_ACTIVE_UNITS, (
        f"Only {n_total} active units — below MIN_TOTAL_ACTIVE_UNITS="
        f"{config.MIN_TOTAL_ACTIVE_UNITS}"
    )

    # ---------------------------------------------------------------------
    # Phase 4: Build canonical UnitIndexMap
    # ---------------------------------------------------------------------
    unit_map = UnitIndexMap.from_mask(mask_3d, grid_shape=config.GRID_DIMS)
    for t in range(config.T):
        assert unit_map.units_per_block[t] >= config.MIN_ACTIVE_UNITS_PER_BLOCK, (
            f"Block {config.TIME_BLOCKS[t][0]} has only "
            f"{unit_map.units_per_block[t]} active units "
            f"(below MIN_ACTIVE_UNITS_PER_BLOCK={config.MIN_ACTIVE_UNITS_PER_BLOCK})"
        )
    print(
        f"[preprocess] n_active_units = {unit_map.n_units} "
        f"(per block: {unit_map.units_per_block.tolist()})",
        flush=True,
    )

    if _should_write("active_mask"):
        save_artifact("active_mask", mask_3d)
    if _should_write("unit_index_map"):
        save_artifact("unit_index_map", unit_map)

    # ---------------------------------------------------------------------
    # Phase 5: Extract per-unit vectors
    # ---------------------------------------------------------------------
    print("[preprocess] Extracting active-unit vectors ...", flush=True)
    D_vec = pickup_3d[mask_3d]
    S_vec = active_taxis_3d[mask_3d]
    demo_flat = demographics_selected.reshape(-1, demographics_selected.shape[-1])

    unit_demographics = np.empty(
        (unit_map.n_units, len(config.DEMOGRAPHIC_FEATURES)),
        dtype=np.float32,
    )
    for i in range(unit_map.n_units):
        flat_cell = unit_map.to_flat_cell(i)
        unit_demographics[i] = demo_flat[flat_cell]

    # Defense-in-depth: Task 11 already enforces this inside precompute_hat_matrices,
    # but surfacing it here fails louder and nearer to the root cause.
    assert np.isfinite(unit_demographics).all(), (
        "Demographics for active units contain NaN — compute_active_mask "
        "should have filtered these cells upstream"
    )

    # ---------------------------------------------------------------------
    # Phase 6: Fit g_0(D)
    # ---------------------------------------------------------------------
    print("[preprocess] Fitting g0(D) ...", flush=True)
    D_clamped = np.maximum(D_vec, config.DEMAND_FLOOR)
    Y_vec = S_vec / D_clamped
    g0_func, g0_diag = fit_g0(D_clamped, Y_vec)
    print(f"[preprocess] g0 diagnostics: {g0_diag}", flush=True)
    if _should_write("g0_power_basis"):
        save_artifact("g0_power_basis", g0_func)

    # ---------------------------------------------------------------------
    # Phase 7: Precompute hat matrices
    # ---------------------------------------------------------------------
    print("[preprocess] Precomputing hat matrices ...", flush=True)
    hat = precompute_hat_matrices(
        demands=D_clamped,
        demographic_features=unit_demographics,
        feature_names=config.DEMOGRAPHIC_FEATURES,
    )
    if _should_write("hat_matrices", include_features=True):
        save_artifact("hat_matrices", hat, include_features=True)

    print("[preprocess] Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess FAMAIL-Temporal raw data.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing cache artifacts.",
    )
    args = parser.parse_args()
    try:
        run(force=args.force)
    except FileNotFoundError as e:
        print(f"[preprocess] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
