"""Load the raw taxi GPS files into a concatenated DataFrame.

Only loads project-internal, trusted files produced by the taxi-GPS data
pipeline. Never deserializes arbitrary external content.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import pandas as pd
import pickle

_COLUMNS = [
    "plate_id", "latitude", "longitude",
    "seconds", "passenger_indicator", "timestamp",
]


def _flatten_driver_records(records_obj) -> list[list]:
    """Handle both flat and nested day-list raw structures."""
    if not isinstance(records_obj, list) or not records_obj:
        return []
    first = records_obj[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        flat: list[list] = []
        for day_list in records_obj:
            if isinstance(day_list, list):
                for rec in day_list:
                    if isinstance(rec, (list, tuple)) and len(rec) >= 6:
                        flat.append(list(rec[:6]))
        return flat
    return [
        list(r[:6]) for r in records_obj
        if isinstance(r, (list, tuple)) and len(r) >= 6
    ]


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load one raw taxi_record_*.pkl file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name}: expected dict keyed by plate_id, got {type(data).__name__}"
        )
    all_records: list[list] = []
    for plate_id, records_obj in data.items():
        for rec in _flatten_driver_records(records_obj):
            rec[0] = str(plate_id) if rec[0] is None else str(rec[0])
            all_records.append(rec)
    if not all_records:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.DataFrame(all_records, columns=_COLUMNS)
    df["plate_id"] = df["plate_id"].astype(str)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["seconds"] = df["seconds"].astype(int)
    df["passenger_indicator"] = df["passenger_indicator"].astype(int)
    df["timestamp"] = df["timestamp"].astype(str)
    return df


def concat_raw_records(paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate multiple raw files into a single DataFrame."""
    dfs = [load_raw_file(p) for p in paths]
    dfs = [d for d in dfs if len(d) > 0]
    if not dfs:
        raise ValueError("concat_raw_records: no non-empty raw files found")
    return pd.concat(dfs, ignore_index=True)
