"""Export per-cell fairness attributions for downstream consumers.

Produces three views of the same data — tuples, long DataFrame, dense
ndarray dict — broadcasting the (cell, time_block) attribution to all
12 time_buckets in the block and to all ``bundle.n_days`` day indices.

Output format is pickle, matching the design doc decision (frozen
2026-04-24) and the existing ``evaluation/persistence.py`` convention
for compatibility with the ``passenger_seeking_trajs_45-800.pkl``
ecosystem. See ``docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md`` for the
design rationale and ``docs/FAIRNESS_DECOMPOSITION_FORMULATION.md``
for the 1/N-shifted per-cell decomposition this tool emits.

Sign convention (verbatim from the design): ``positive_is_fair``.
The sum of either attribution column across all active (cell, time_block)
units equals the overall F-metric (F_spatial or F_causal). Per-cell
values are signed real numbers, NOT bounded in [0, 1] — only the global
metric is in [0, 1]. Consumers must clamp/normalize themselves if their
loss requires bounded scalars.

CLI:
    python -m famail_temporal.evaluation.export_fairness_attributions \\
        [--name NAME] [--max-trajectories N] [--max-drivers N]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pickle as _pkl  # see module docstring for the design-frozen format choice
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.persistence import _git_sha, _git_dirty
from famail_temporal.fairness.causal import compute_fcausal_from_compact
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.fairness.spatial import compute_fspatial


SCHEMA_VERSION = "1.0.0"
SIGN_CONVENTION = "positive_is_fair"
BUCKETS_PER_DAY = config.N_TIME_BUCKETS  # 288
BUCKETS_PER_HOUR = 12  # 5-minute resolution


@dataclass(frozen=True)
class ExportData:
    """Dense per-(cell, time_block) views + provenance metadata.

    Attribution arrays carry NaN at inactive cells. The long/tuples
    consumers are produced by broadcasting these (gx, gy, T) arrays
    along the time_bucket and day axes.
    """
    spatial_attribution: np.ndarray   # (gx, gy, T) float32, NaN on inactive
    causal_attribution: np.ndarray    # (gx, gy, T) float32, NaN on inactive
    active_mask: np.ndarray           # (gx, gy, T) bool
    demand_D: np.ndarray              # (gx, gy, T) float32, NaN on inactive
    supply_S: np.ndarray              # (gx, gy, T) float32, NaN on inactive
    service_rate_Y: np.ndarray        # (gx, gy, T) float32, NaN on inactive
    n_days: int
    metadata: Dict[str, Any]


def _scalar_F_metrics(bundle: DataBundle) -> Tuple[float, float]:
    """Compute pooled F_spatial and F_causal on the bundle's active units."""
    import torch
    mask = bundle.mask_3d
    pickup_N = torch.from_numpy(bundle.pickup_3d[mask]).float()
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[mask]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[mask]).float()
    f_spatial, _ = compute_fspatial(pickup_N, dropoff_N, active_N)

    D_clamped = torch.clamp(pickup_N, min=config.DEMAND_FLOOR)
    g0_D = torch.from_numpy(
        np.asarray(bundle.g0_func(D_clamped.numpy()), dtype=np.float32)
    )
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    f_causal, _ = compute_fcausal_from_compact(
        D_clamped, active_N, g0_D, tensors["X_demo"], tensors["XtX_inv"],
    )
    return float(f_spatial), float(f_causal)


def _dense_DSY(bundle: DataBundle) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense (gx, gy, T) D/S/Y views. Inactive cells are NaN."""
    mask = bundle.mask_3d
    D = np.where(mask, bundle.pickup_3d, np.nan).astype(np.float32)
    S = np.where(mask, bundle.active_taxis_3d, np.nan).astype(np.float32)
    D_clamped = np.maximum(bundle.pickup_3d, config.DEMAND_FLOOR)
    Y_dense = bundle.active_taxis_3d / D_clamped
    Y = np.where(mask, Y_dense, np.nan).astype(np.float32)
    return D, S, Y


def _processing_metadata_snapshot() -> Optional[Dict[str, Any]]:
    """Return a slim provenance snapshot of source_generation's processing_metadata.

    The full sidecar contains a per-driver removal_summary that's tens of MB and
    not useful to export consumers. We keep only the high-level provenance
    fields (git SHA, n_days, bounds, source-side config_snapshot) and drop the
    detailed removal/per-driver blobs. Consumers wanting the full sidecar can
    read it directly from the source_data dir at the recorded git_sha.
    """
    path = config.SOURCE_DATA_DIR / "processing_metadata.json"
    if not path.exists():
        return None
    try:
        full = json.loads(path.read_text())
    except Exception:
        return None
    keep = ("n_days", "bounds", "git_sha", "config_snapshot")
    slim = {k: full[k] for k in keep if k in full}
    # Replace the heavy removal_summary with just the top-level counts (if any).
    rs = full.get("removal_summary")
    if isinstance(rs, dict):
        slim["removal_summary_counts"] = {
            k: v for k, v in rs.items() if not isinstance(v, (list, dict))
        }
    return slim


def _config_snapshot() -> Dict[str, Any]:
    keys = [
        "GRID_DIMS", "T", "N_TIME_BUCKETS",
        "DEMAND_FLOOR", "SUPPLY_FLOOR", "ACTIVE_SUPPLY_THRESHOLD",
        "DEMOGRAPHIC_FEATURES",
    ]
    snap: Dict[str, Any] = {}
    for k in keys:
        if hasattr(config, k):
            snap[k] = getattr(config, k)
    snap["TIME_BLOCKS"] = [list(b) for b in config.TIME_BLOCKS]
    return snap


def build_export_data(bundle: DataBundle) -> ExportData:
    """Assemble dense (gx, gy, T) attribution + D/S/Y views from a bundle."""
    grid = build_fairness_grid(bundle)  # (gx, gy, T, 4); NaN on inactive
    spatial_attr = grid[..., 0].astype(np.float32)
    causal_attr = grid[..., 1].astype(np.float32)
    D, S, Y = _dense_DSY(bundle)
    f_spatial, f_causal = _scalar_F_metrics(bundle)

    units_per_block = bundle.unit_map.units_per_block.tolist()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "famail_git_sha": _git_sha(),
        "famail_git_dirty": _git_dirty(),
        "source_data_processing_metadata": _processing_metadata_snapshot(),
        "config_snapshot": _config_snapshot(),
        "overall_F_spatial": float(f_spatial),
        "overall_F_causal": float(f_causal),
        "sign_convention": SIGN_CONVENTION,
        "n_active_cells_per_block": units_per_block,
        "n_days": int(bundle.n_days),
    }
    return ExportData(
        spatial_attribution=spatial_attr,
        causal_attribution=causal_attr,
        active_mask=bundle.mask_3d.copy(),
        demand_D=D, supply_S=S, service_rate_Y=Y,
        n_days=int(bundle.n_days),
        metadata=metadata,
    )


def _broadcast_to_buckets(
    cell_block_array: np.ndarray, n_days: int,
) -> np.ndarray:
    """Broadcast (gx, gy, T) → (gx, gy, BUCKETS_PER_DAY, n_days).

    Each time_block covers ``BUCKETS_PER_HOUR`` consecutive 1-indexed
    time_buckets; the value repeats identically across them. Then the
    bucket-axis result is repeated across all ``n_days`` day indices.
    """
    gx, gy, T = cell_block_array.shape
    if T * BUCKETS_PER_HOUR != BUCKETS_PER_DAY:
        raise ValueError(
            f"T * 12 ({T * BUCKETS_PER_HOUR}) != BUCKETS_PER_DAY "
            f"({BUCKETS_PER_DAY}); broadcasting assumes hourly blocks."
        )
    by_bucket = np.repeat(cell_block_array, BUCKETS_PER_HOUR, axis=2)
    return np.broadcast_to(
        by_bucket[..., None], (gx, gy, BUCKETS_PER_DAY, n_days),
    ).copy()


def write_dense_pkl(data: ExportData, path: Path) -> Path:
    """Write the dense (gx, gy, T) view at the BLOCK level (not bucket)."""
    payload = {
        "spatial": data.spatial_attribution,
        "causal": data.causal_attribution,
        "active_mask": data.active_mask,
        "D": data.demand_D,
        "S": data.supply_S,
        "Y": data.service_rate_Y,
        "metadata": data.metadata,
    }
    path.write_bytes(_pkl.dumps(payload, protocol=4))
    return path


_TUPLE_COLUMNS = (
    "x_grid", "y_grid", "time_bucket", "day",
    "spatial_fairness_attribution", "causal_fairness_attribution",
    "is_active", "demand_D", "supply_S", "service_rate_Y",
)


def _row_iter(data: ExportData):
    """Yield row-tuples matching ``_TUPLE_COLUMNS`` over the broadcast grid.

    Coordinates emitted on disk are 1-indexed per the design row schema.
    """
    n_days = data.n_days
    spatial_b = _broadcast_to_buckets(data.spatial_attribution, n_days)
    causal_b = _broadcast_to_buckets(data.causal_attribution, n_days)
    active_b = _broadcast_to_buckets(
        data.active_mask.astype(np.uint8), n_days,
    ).astype(bool)
    D_b = _broadcast_to_buckets(data.demand_D, n_days)
    S_b = _broadcast_to_buckets(data.supply_S, n_days)
    Y_b = _broadcast_to_buckets(data.service_rate_Y, n_days)
    gx, gy, n_buckets, _ = spatial_b.shape
    for x in range(gx):
        for y in range(gy):
            for tb_zero in range(n_buckets):
                for d_zero in range(n_days):
                    yield (
                        x + 1, y + 1, tb_zero + 1, d_zero + 1,
                        float(spatial_b[x, y, tb_zero, d_zero]),
                        float(causal_b[x, y, tb_zero, d_zero]),
                        bool(active_b[x, y, tb_zero, d_zero]),
                        float(D_b[x, y, tb_zero, d_zero]),
                        float(S_b[x, y, tb_zero, d_zero]),
                        float(Y_b[x, y, tb_zero, d_zero]),
                    )


def write_tuples_pkl(data: ExportData, path: Path) -> Path:
    """Write a list of row-tuples + metadata preamble."""
    rows = list(_row_iter(data))
    payload = {
        "metadata": data.metadata,
        "columns": list(_TUPLE_COLUMNS),
        "rows": rows,
    }
    path.write_bytes(_pkl.dumps(payload, protocol=4))
    return path


def write_long_pkl(data: ExportData, path: Path) -> Path:
    """Write the row-tuples as a pandas DataFrame pickle."""
    import pandas as pd
    df = pd.DataFrame(_row_iter(data), columns=list(_TUPLE_COLUMNS))
    payload = {"metadata": data.metadata, "dataframe": df}
    path.write_bytes(_pkl.dumps(payload, protocol=4))
    return path


def write_metadata_json(data: ExportData, path: Path) -> Path:
    """Write the metadata sidecar as JSON (humans + GAN trainers)."""
    path.write_text(json.dumps(data.metadata, indent=2, default=str))
    return path


_README_TEMPLATE = """# FAMAIL Fairness Attribution Export

Schema version: **{schema_version}**
Generated: **{generated_at}**
Sign convention: **{sign_convention}**

## TL;DR

Per-cell fairness attributions for the FAMAIL spatial-temporal grid.
Use **positive value = more fair** as your reward signal. Attribution
values are **signed** real numbers — clamp to [0, 1] only if your
loss function requires it.

## Sign convention

Per-cell values sum to the overall F-metric (1/N-shifted decomposition):

    Sum over active cells  spatial_fairness_attribution  ==  F_spatial
    Sum over active cells  causal_fairness_attribution   ==  F_causal

- positive  -> cell contributes more than the 1/N baseline to fairness (good)
- ~ 0       -> cell at the negative-fair / anti-fair boundary
- negative  -> cell drags fairness below baseline (drag cell)

See `docs/FAIRNESS_DECOMPOSITION_FORMULATION.md` for the full formulation.

## Overall vs per-cell scale

The OVERALL metrics are bounded:

    F_spatial in [0, 1]    F_causal in [0, 1]    higher = more fair

Per-cell attributions are NOT bounded in [0, 1]. They are signed
real numbers whose sum equals the overall metric. If your loss
requires a bounded per-cell scalar, normalize on your side (no
universal choice fits every consumer).

This export's overall metrics:

- F_spatial = **{overall_F_spatial:.6f}**
- F_causal  = **{overall_F_causal:.6f}**

## Granularity

- **Time.** Fairness is computed per `(x, y, time_block)`. The export
  broadcasts each block's attribution identically across all 12 of
  the 5-minute time_buckets in that block.
- **Day.** Fairness is computed pooled across days. The export
  broadcasts the same (cell, block) attribution to all
  `n_days = {n_days}` day indices.
- **Active vs inactive.** Every (x, y, time_bucket, day) appears in
  the long/tuples outputs. Inactive cells (no supply, out of bounds,
  or NaN demographics) have `is_active = False` and NaN attributions.

## File reference

| File | Schema |
|---|---|
| `fairness_attribution_dense.pkl` | dict of (gx, gy, T) ndarrays at the BLOCK level + metadata; fastest tensor lookup |
| `fairness_attribution_tuples.pkl` | metadata + list of row-tuples at (cell, bucket, day) granularity |
| `fairness_attribution_long.pkl`   | metadata + pandas DataFrame, same row-tuples |
| `metadata.json` | provenance sidecar (also embedded in each .pkl) |

### Row-level schema (long / tuples)

| Column | Type | Range | Description |
|---|---|---|---|
| `x_grid` | int | [1, {gx}] | Grid-cell x coordinate, 1-indexed |
| `y_grid` | int | [1, {gy}] | Grid-cell y coordinate, 1-indexed |
| `time_bucket` | int | [1, {n_buckets}] | 5-minute time bucket, 1-indexed |
| `day` | int | [1, {n_days}] | Day index, 1-indexed |
| `spatial_fairness_attribution` | float (NaN if inactive) | signed | Per-cell contribution to `F_spatial`; positive = more fair |
| `causal_fairness_attribution`  | float (NaN if inactive) | signed | Per-cell contribution to `F_causal`; positive = more fair |
| `is_active` | bool | -- | Whether this cell is in the fairness audit |
| `demand_D` | float (NaN if inactive) | >= 0 | Mean hourly pickups in (cell, time_block) |
| `supply_S` | float (NaN if inactive) | >= 0 | Mean hourly active taxis in (cell, time_block) |
| `service_rate_Y` | float (NaN if inactive) | > 0 | `S / max(D, DEMAND_FLOOR)` |

## Example lookup

```python
import pickle

# Dense (block-level) -- fastest in a training loop
with open("fairness_attribution_dense.pkl", "rb") as f:
    dense = pickle.load(f)
# dense["spatial"][x, y, t_block]   # NaN if inactive
# dense["causal"][x, y, t_block]
# dense["active_mask"][x, y, t_block]

# Long (DataFrame) -- best for filtering / pandas analysis
with open("fairness_attribution_long.pkl", "rb") as f:
    payload = pickle.load(f)
df = payload["dataframe"]
fair_cells = df[df["is_active"] & (df["spatial_fairness_attribution"] > 0)]

# Tuples -- iterator-friendly, no-dependency consumer
with open("fairness_attribution_tuples.pkl", "rb") as f:
    payload = pickle.load(f)
columns = payload["columns"]
for row in payload["rows"]:
    record = dict(zip(columns, row))
    if record["is_active"]:
        ...
```

## Reproducibility

- `famail_git_sha`: **{famail_git_sha}** (`dirty={famail_git_dirty}`)
- Source-data processing metadata embedded in `metadata.json` under
  `source_data_processing_metadata` (or `null` if the sidecar was not
  available at export time).
- Config snapshot in `metadata.json` under `config_snapshot`.

## How to use this export

For a prescriptive how-to on plugging this export into a GAN, GAIL, or
generic offline-RL training loop — including loading patterns, the
sign convention, the broadcast trap, pitfalls, and sanity checks —
see [`../USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`](../USING_FAIRNESS_ATTRIBUTION_EXPORTS.md).

## Contact

Methodology questions:
`docs/F_CAUSAL_METHODOLOGY_NOTES.md` and
`docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`.
"""


def write_readme(data: ExportData, path: Path) -> Path:
    gx, gy, _T = data.spatial_attribution.shape
    text = _README_TEMPLATE.format(
        schema_version=SCHEMA_VERSION,
        generated_at=data.metadata["generated_at"],
        sign_convention=SIGN_CONVENTION,
        overall_F_spatial=data.metadata["overall_F_spatial"],
        overall_F_causal=data.metadata["overall_F_causal"],
        n_days=data.n_days,
        gx=gx, gy=gy,
        n_buckets=BUCKETS_PER_DAY,
        famail_git_sha=data.metadata["famail_git_sha"],
        famail_git_dirty=data.metadata["famail_git_dirty"],
    )
    path.write_text(text)
    return path


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name).strip("-")


def _generate_export_id(name: Optional[str]) -> str:
    timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if name:
        return f"{timestamp}_{_slugify(name)}"
    return timestamp


def export(
    bundle: DataBundle,
    output_root: Path,
    name: Optional[str] = None,
) -> Path:
    """Write all four artifacts (dense / tuples / long / metadata) + README.

    Returns the export directory path.
    """
    export_id = _generate_export_id(name)
    out_dir = output_root / export_id
    out_dir.mkdir(parents=True, exist_ok=False)
    data = build_export_data(bundle)
    write_dense_pkl(data, out_dir / "fairness_attribution_dense.pkl")
    write_tuples_pkl(data, out_dir / "fairness_attribution_tuples.pkl")
    write_long_pkl(data, out_dir / "fairness_attribution_long.pkl")
    write_metadata_json(data, out_dir / "metadata.json")
    write_readme(data, out_dir / "README.md")
    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="famail_temporal.evaluation.export_fairness_attributions",
        description=(
            "Export per-cell fairness attributions for downstream consumers "
            "(GAN/GAIL training, baseline-GAN comparison)."
        ),
    )
    p.add_argument("--name", default=None, help="Optional run-name slug.")
    p.add_argument(
        "--max-trajectories", type=int, default=None,
        help="Cap trajectories loaded from the cache (smoke tests).",
    )
    p.add_argument(
        "--max-drivers", type=int, default=None,
        help="Cap drivers loaded from the cache (smoke tests).",
    )
    p.add_argument(
        "--output-root", default=None, type=str,
        help="Override the output root (default: <package_root>/exports/).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    bundle = DataBundle.load(
        max_trajectories=args.max_trajectories,
        max_drivers=args.max_drivers,
    )
    output_root = (
        Path(args.output_root) if args.output_root
        else Path(config.PACKAGE_ROOT) / "exports"
    )
    out_dir = export(bundle, output_root=output_root, name=args.name)
    print(f"[export] export_dir   = {out_dir}")
    print(f"[export] dense        = {out_dir / 'fairness_attribution_dense.pkl'}")
    print(f"[export] tuples       = {out_dir / 'fairness_attribution_tuples.pkl'}")
    print(f"[export] long (df)    = {out_dir / 'fairness_attribution_long.pkl'}")
    print(f"[export] metadata     = {out_dir / 'metadata.json'}")
    print(f"[export] README       = {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
