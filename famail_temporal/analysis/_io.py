from __future__ import annotations
import json
from pathlib import Path


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def editor_metrics(run_dir) -> dict:
    return read_json(Path(run_dir) / "metrics.json")


def processing_metadata(source_dir) -> dict:
    return read_json(Path(source_dir) / "processing_metadata.json")
