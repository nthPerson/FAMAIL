"""Write all tool outputs: 9 output files + driver mapping + metadata JSON."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
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
    calendar_day_map: Path
    driver_mapping: Path
    metadata: Path


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
    path: Path, *, raw, normalized, mean, std, feature_names: list[str],
    n_features: int, drivers_mapping: dict,
) -> None:
    """Store RAW features in ``features`` and NORMALIZED in
    ``features_normalized``.

    Downstream consumers split along this seam:
      - ``famail_temporal/data/loader.py`` reads ``features_normalized``
        (it wants ready-to-use z-scored vectors).
      - ``discriminator/multi_stream/dataset_generation/generation.py``
        reads ``features`` and applies z-score normalization itself using
        the stored ``mean`` / ``std``. If ``features`` held already-normalized
        values, that step would double-normalize and produce garbage training
        data — see the design spec for the regression test that catches this.
    """
    raw_dict = {
        int(idx): raw[int(idx)].astype(float)
        for idx in drivers_mapping["idx_to_plate"].keys()
    }
    normalized_dict = {
        int(idx): normalized[int(idx)].astype(float)
        for idx in drivers_mapping["idx_to_plate"].keys()
    }
    bundle = {
        "features": raw_dict,
        "features_normalized": normalized_dict,
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
        calendar_day_map=out_dir / config.OUT_CALENDAR_DAY_MAP,
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
        raw=ms_profile["raw"],
        normalized=ms_profile["normalized"],
        mean=ms_profile["mean"], std=ms_profile["std"],
        feature_names=ms_profile["feature_names"],
        n_features=config.N_PROFILE_FEATURES,
        drivers_mapping=driver_mapping,
    )
    _pickle_write(paths.ms_seeking_days, ms_calendars["seeking"])
    _pickle_write(paths.ms_driving_days, ms_calendars["driving"])
    _pickle_write(paths.calendar_day_map, ms_calendars["calendar_day_map"])
    _pickle_write(paths.driver_mapping, driver_mapping)
    # Compute byte-level fingerprints of all 10 data .pkl outputs (not metadata itself)
    pkl_paths = [
        paths.pickup_dropoff, paths.active_taxis, paths.passenger_seeking,
        paths.ms_seeking, paths.ms_driving, paths.ms_profile,
        paths.ms_seeking_days, paths.ms_driving_days,
        paths.calendar_day_map, paths.driver_mapping,
    ]
    data_sha256 = {p.name: _sha256_file(p) for p in pkl_paths}
    # Build a new dict so we don't mutate the caller's metadata_extras
    metadata_extras = {**metadata_extras, "data_sha256": data_sha256}
    write_metadata_json(paths.metadata, removal_summary, metadata_extras)
    return paths
