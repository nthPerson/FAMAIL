"""Cache I/O helpers — all cache artifacts go through these functions."""

from __future__ import annotations
import pickle as _pkl
from pathlib import Path
from typing import Any

from famail_temporal import config


def cache_path(artifact_name: str, include_features: bool = False) -> Path:
    """Build the cache file path with config-encoded suffix."""
    suffix = config.cache_suffix(include_features=include_features)
    return config.CACHE_DIR / f"{artifact_name}_{suffix}.pkl"


def save_artifact(artifact_name: str, data: Any, include_features: bool = False) -> Path:
    """Pickle-serialize `data` into the cache path for `artifact_name`."""
    path = cache_path(artifact_name, include_features=include_features)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        _pkl.dump(data, f)
    return path


def load_artifact(artifact_name: str, include_features: bool = False) -> Any:
    """Load the cached artifact pickle — fail loud with a remediation hint."""
    path = cache_path(artifact_name, include_features=include_features)
    if not path.exists():
        raise FileNotFoundError(
            f"Cache artifact missing: {path}. "
            f"Run: python -m famail_temporal.preprocess"
        )
    with open(path, "rb") as f:
        return _pkl.load(f)


def load_raw(filename: str) -> Any:
    """Load a source-data .pkl from the source_data directory."""
    path = config.SOURCE_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Source data missing: {path}. See source_data/README.md."
        )
    with open(path, "rb") as f:
        return _pkl.load(f)
