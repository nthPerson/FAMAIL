"""Write all tool outputs: 8 output files + driver mapping + metadata JSON."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import pickle

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.removal import RemovalSummary


@dataclass(frozen=True)
class OutputPaths:
    pickup_dropoff: Path
    active_taxis: Path
    passenger_seeking: Path
    ms_seeking: Path
    ms_driving: Path
    ms_profile: Path
    ms_seeking_days: Path
    ms_driving_days: Path
    driver_mapping: Path
    metadata: Path


def _pickle_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_active_taxis_bundle(
    path: Path, counts: dict, stats: dict, config_snapshot: dict,
) -> None:
    bundle = {
        "data": counts,
        "stats": stats,
        "config": config_snapshot,
        "version": config.OUTPUT_FORMAT_VERSION,
    }
    _pickle_write(path, bundle)


def write_profile_bundle(
    path: Path, normalized, mean, std, feature_names: list[str],
    n_features: int, drivers_mapping: dict,
) -> None:
    features = {
        int(idx): normalized[int(idx)].astype(float)
        for idx in drivers_mapping["idx_to_plate"].keys()
    }
    bundle = {
        "features": features,
        "features_normalized": features,
        "feature_names": feature_names,
        "normalization": {"mean": mean.astype(float), "std": std.astype(float)},
        "n_features": n_features,
    }
    _pickle_write(path, bundle)


def write_metadata_json(
    path: Path, removal_summary: RemovalSummary, extras: dict,
) -> None:
    removals_dict_list = [r.to_dict() for r in removal_summary.removals]
    metadata = dict(extras)
    metadata["removal_summary"] = {
        "total_seeking_extracted": removal_summary.total_seeking_extracted,
        "total_driving_extracted": removal_summary.total_driving_extracted,
        "total_extracted": removal_summary.total_extracted(),
        "n_removed": len(removal_summary.removals),
        "removal_rate": removal_summary.removal_rate(),
        "counts_by_category": removal_summary.counts_by_category(),
        "removals": removals_dict_list,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def write_all_outputs(
    out_dir: Path,
    pickup_dropoff: dict,
    active_taxis: dict,
    passenger_seeking_trajs: dict,
    ms_seeking: dict,
    ms_driving: dict,
    ms_profile: dict,
    ms_calendars: dict,
    driver_mapping: dict,
    removal_summary: RemovalSummary,
    metadata_extras: dict,
) -> OutputPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = OutputPaths(
        pickup_dropoff=out_dir / config.OUT_PICKUP_DROPOFF,
        active_taxis=out_dir / config.OUT_ACTIVE_TAXIS,
        passenger_seeking=out_dir / config.OUT_PASSENGER_SEEKING,
        ms_seeking=out_dir / config.OUT_MS_SEEKING,
        ms_driving=out_dir / config.OUT_MS_DRIVING,
        ms_profile=out_dir / config.OUT_MS_PROFILE,
        ms_seeking_days=out_dir / config.OUT_MS_SEEKING_DAYS,
        ms_driving_days=out_dir / config.OUT_MS_DRIVING_DAYS,
        driver_mapping=out_dir / config.OUT_DRIVER_MAPPING,
        metadata=out_dir / config.OUT_METADATA,
    )
    _pickle_write(paths.pickup_dropoff, pickup_dropoff)
    write_active_taxis_bundle(
        paths.active_taxis, active_taxis,
        stats={"n_entries": len(active_taxis)},
        config_snapshot={
            "neighborhood_dims": config.NEIGHBORHOOD_SIZE,
            "period_type": "hourly",
        },
    )
    _pickle_write(paths.passenger_seeking, passenger_seeking_trajs)
    _pickle_write(paths.ms_seeking, ms_seeking)
    _pickle_write(paths.ms_driving, ms_driving)
    write_profile_bundle(
        paths.ms_profile,
        ms_profile["normalized"], ms_profile["mean"], ms_profile["std"],
        ms_profile["feature_names"], config.N_PROFILE_FEATURES, driver_mapping,
    )
    _pickle_write(paths.ms_seeking_days, ms_calendars["seeking"])
    _pickle_write(paths.ms_driving_days, ms_calendars["driving"])
    _pickle_write(paths.driver_mapping, driver_mapping)
    write_metadata_json(paths.metadata, removal_summary, metadata_extras)
    return paths
