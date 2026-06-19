"""District geometry: loading, centroids, orientation guard, boundary segments.

Orientation is the project's historical pain point. The canonical convention
is array[x_grid(row, 0=South..47=North), y_grid(col, 0=West..89=East)] displayed
with y_grid horizontal (W->E) and x_grid vertical with SOUTH at the bottom.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DistrictGeometry:
    district_id_grid: np.ndarray   # (48,90) int8, -1 = non-Shenzhen
    valid_mask: np.ndarray         # (48,90) bool
    district_names: list           # len 10
    boundary_x: np.ndarray         # 1-D float, display x (=col), NaN-separated
    boundary_y: np.ndarray         # 1-D float, display y (=row), NaN-separated


def compute_boundary_segments(region_label_grid: np.ndarray):
    """Line segments along edges between adjacent cells with differing labels.

    Returns (xs, ys) in DISPLAY coords (x=col=y_grid, y=row=x_grid). Each segment
    is two finite points followed by NaN, so a single Plotly/Matplotlib line trace
    renders all segments with gaps between them.  When callers pass
    ``district_id_grid`` (which uses −1 for non-Shenzhen cells), the segments
    include the Shenzhen outer perimeter — edges between valid district cells and
    −1 non-Shenzhen cells — in addition to inter-district edges.
    """
    g = region_label_grid
    rows, cols = g.shape
    xs: list[float] = []
    ys: list[float] = []
    # Vertical edges between (i, j) and (i, j+1): x = j+0.5, y in [i-0.5, i+0.5]
    for i in range(rows):
        for j in range(cols - 1):
            if g[i, j] != g[i, j + 1]:
                xs += [j + 0.5, j + 0.5, np.nan]
                ys += [i - 0.5, i + 0.5, np.nan]
    # Horizontal edges between (i, j) and (i+1, j): y = i+0.5, x in [j-0.5, j+0.5]
    for i in range(rows - 1):
        for j in range(cols):
            if g[i, j] != g[i + 1, j]:
                xs += [j - 0.5, j + 0.5, np.nan]
                ys += [i + 0.5, i + 0.5, np.nan]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def district_centroids(district_id_grid: np.ndarray, district_names: list):
    """name -> (mean row=x_grid, mean col=y_grid)."""
    out = {}
    for idx, name in enumerate(district_names):
        rows, cols = np.where(district_id_grid == idx)
        if rows.size:
            out[name] = (float(rows.mean()), float(cols.mean()))
    return out


def assert_canonical_orientation(district_id_grid: np.ndarray, district_names: list) -> None:
    """Fail loudly if the grid is flipped/mirrored from the verified convention."""
    c = district_centroids(district_id_grid, district_names)
    nan_row, nan_col = c["Nanshan"]
    bao_row, bao_col = c["Bao'an"]
    dap_row, dap_col = c["Dapeng"]
    gua_row, gua_col = c["Guangming"]
    if not (nan_row < 20):
        raise AssertionError(f"Nanshan should be south (row<20), got {nan_row:.1f}")
    if not (nan_col < 25):
        raise AssertionError(f"Nanshan should be west (col<25), got {nan_col:.1f}")
    if not (bao_col < 18):
        raise AssertionError(f"Bao'an should be far west (col<18), got {bao_col:.1f}")
    if not (dap_col > 65):
        raise AssertionError(f"Dapeng should be far east (col>65), got {dap_col:.1f}")
    if not (gua_row > 30):
        raise AssertionError(f"Guangming should be north (row>30), got {gua_row:.1f}")


def load_district_geometry(mapping_path: Optional[Path] = None) -> DistrictGeometry:
    if mapping_path is None:
        from famail_temporal import config
        mapping_path = Path(config.SOURCE_DATA_DIR) / "grid_to_district_mapping.pkl"
    # pickle is safe here: mapping_path is a project-internal artifact generated
    # by famail_temporal's own preprocessing pipeline, never user-supplied.
    with open(mapping_path, "rb") as f:
        m = pickle.load(f)
    did = np.asarray(m["district_id_grid"]).astype(np.int8)
    valid = np.asarray(m["valid_mask"]).astype(bool)
    names = list(m["district_names"])
    bx, by = compute_boundary_segments(did)
    return DistrictGeometry(
        district_id_grid=did, valid_mask=valid, district_names=names,
        boundary_x=bx, boundary_y=by,
    )
