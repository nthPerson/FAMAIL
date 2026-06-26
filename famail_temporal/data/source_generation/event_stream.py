"""Build the single enriched event-stream DataFrame — the load-bearing
intermediate representation every view derives from."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.quantization import (
    GlobalBounds, compute_global_bounds, gps_to_grid,
    seconds_to_time_bucket, seconds_to_hour, timestamp_to_day,
)
from famail_temporal.data.source_generation.raw_loader import concat_raw_records
from famail_temporal.data.source_generation.stuck_gps import filter_stuck_gps_sinks
from famail_temporal.data.source_generation.transitions import (
    add_transition_columns, assign_segment_ids,
)


@dataclass(frozen=True)
class EventStream:
    df: pd.DataFrame
    bounds: GlobalBounds
    n_days: int
    driver_calendar_days: dict[str, set[int]]
    sink_audit: dict = field(default_factory=dict)


def _load_quantized_sorted(raw_dir: Path) -> tuple[pd.DataFrame, GlobalBounds]:
    """Load raw records → quantize (grid + time) → weekday-filter → sort.

    Returns the sorted-but-pre-transition DataFrame and the GlobalBounds.
    Does NOT add transition columns; that is the caller's responsibility so the
    stuck-GPS filter can run on the pre-transition df.
    """
    paths = [raw_dir / name for name in config.RAW_INPUT_FILENAMES]
    df = concat_raw_records(paths)

    bounds = compute_global_bounds(df["latitude"], df["longitude"])

    xs, ys = gps_to_grid(df["latitude"].values, df["longitude"].values, bounds)
    df["x_grid"] = xs
    df["y_grid"] = ys
    df["time_bucket"] = seconds_to_time_bucket(df["seconds"].values)
    df["hour"] = seconds_to_hour(df["seconds"].values)

    df["day_index"] = df["timestamp"].map(timestamp_to_day)
    df["calendar_date"] = df["timestamp"].str[:10]
    df = df.dropna(subset=["day_index"]).reset_index(drop=True)
    df["day_index"] = df["day_index"].astype(int)

    df = df.sort_values(["plate_id", "timestamp"], kind="stable").reset_index(drop=True)

    return df, bounds


def build_event_stream(raw_dir: Path, *, apply_sink_filter: bool = True) -> EventStream:
    """Load raw → quantize → weekday-filter → sort → (optionally) filter stuck-GPS
    sinks → detect transitions."""
    df, bounds = _load_quantized_sorted(raw_dir)

    sink_audit: dict = {}
    if apply_sink_filter:
        df, sink_audit = filter_stuck_gps_sinks(
            df,
            min_pickups=config.STUCK_GPS_MIN_PICKUPS,
            coord_dominance=config.STUCK_GPS_COORD_DOMINANCE,
            coord_precision=config.STUCK_GPS_COORD_PRECISION,
            expected_cells=config.STUCK_GPS_EXPECTED_CELLS,
            drop=config.STUCK_GPS_DROP,
        )

    df = add_transition_columns(df)
    df = assign_segment_ids(df)

    n_days = df["calendar_date"].nunique()
    driver_calendar_days: dict[str, set[int]] = (
        df.groupby("plate_id")["day_index"].apply(lambda s: set(int(d) for d in s)).to_dict()
    )

    return EventStream(
        df=df, bounds=bounds, n_days=n_days,
        driver_calendar_days=driver_calendar_days,
        sink_audit=sink_audit,
    )
