"""Per-sink F_spatial decomposition (E23).

Pure function: sink_spatial_contributions
CLI driver: decompose  [DEFERRED EXECUTION — do not run while the experiment
sequence is live; see plan Task 5 for the safe-swap procedure]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

# The 10 calibrated stuck-GPS sink cells (1-indexed, matching the pipeline grid).
# Headline sink is (29, 53).
DEFAULT_SINK_CELLS: List[Tuple[int, int]] = [
    (29, 53), (18, 40), (2, 85), (2, 84),
    (3, 85), (3, 84), (2, 86), (3, 86),
    (2, 83), (3, 83),
]


def sink_spatial_contributions(
    dense_spatial: np.ndarray,
    active_mask: np.ndarray,
    sink_cells_1idx: List[Tuple[int, int]],
) -> dict:
    """Compute each sink cell's contribution to F_spatial from the dense attribution.

    Parameters
    ----------
    dense_spatial : np.ndarray, shape (gx, gy, T)
        Spatial channel (channel 0) of the fairness_attribution_dense array.
        Values represent per-cell/per-time-block spatial attribution αᵢ.
    active_mask : np.ndarray, shape (gx, gy, T), dtype bool
        True where a cell/t-block is active (has non-zero demand).
    sink_cells_1idx : list of (x, y) tuples
        Sink cells in 1-indexed grid coordinates (pipeline convention).
        Converted to 0-indexed internally via −1.

    Returns
    -------
    dict with keys:
        "per_sink"  : dict mapping "(x, y)" -> float summed spatial contrib
        "total"     : float sum of all per-sink contributions
    """
    per_sink: dict = {}
    for cell in sink_cells_1idx:
        x1, y1 = cell
        x0, y0 = x1 - 1, y1 - 1
        cell_mask = active_mask[x0, y0, :]
        contrib = float(np.nansum(dense_spatial[x0, y0, :][cell_mask]))
        per_sink[str(cell)] = contrib
    total = float(sum(per_sink.values()))
    return {"per_sink": per_sink, "total": total}


# ---------------------------------------------------------------------------
# CLI driver (DEFERRED — do NOT execute while baaigffdf is live)
# ---------------------------------------------------------------------------

def _load_dense(pkl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load fairness_attribution_dense.pkl -> (spatial_array, active_mask).

    pickle is safe here: fairness_attribution_dense.pkl is a project-internal
    artifact produced by famail_temporal's own export_fairness_attributions
    pipeline (numpy arrays only), never user-supplied external data.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["spatial"], data["active_mask"]


def decompose(
    dirty_dense_pkl: Path,
    clean_dense_pkl: Path,
    sink_cells: List[Tuple[int, int]],
    out_dir: Path,
) -> Path:
    """Compute per-sink F_spatial contribution (dirty & clean) and their delta.

    Writes ``sink_f_spatial_decomposition.json`` to out_dir.

    Returns the output path.
    """
    dirty_spatial, dirty_mask = _load_dense(dirty_dense_pkl)
    clean_spatial, clean_mask = _load_dense(clean_dense_pkl)

    dirty_contribs = sink_spatial_contributions(dirty_spatial, dirty_mask, sink_cells)
    clean_contribs = sink_spatial_contributions(clean_spatial, clean_mask, sink_cells)

    # Compute per-sink delta and share of total F_spatial shift
    total_shift = clean_contribs["total"] - dirty_contribs["total"]
    per_sink_results: dict = {}
    for cell_key in dirty_contribs["per_sink"]:
        dc = dirty_contribs["per_sink"][cell_key]
        cc = clean_contribs["per_sink"].get(cell_key, 0.0)
        delta = cc - dc
        share = (delta / total_shift) if abs(total_shift) > 1e-12 else None
        per_sink_results[cell_key] = {
            "dirty_contrib": dc,
            "clean_contrib": cc,
            "delta": delta,
            "share_of_total_shift": share,
        }

    result = {
        "dirty_total": dirty_contribs["total"],
        "clean_total": clean_contribs["total"],
        "total_shift": total_shift,
        "per_sink": per_sink_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sink_f_spatial_decomposition.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.analysis.sink_decomposition",
        description=(
            "DEFERRED EXECUTION: run only after the experiment sequence finishes "
            "and you have exported fairness_attribution_dense.pkl for both "
            "dirty and clean bundles."
        ),
    )
    ap.add_argument("--dirty-export", type=Path, required=True,
                    help="Dir containing dirty fairness_attribution_dense.pkl")
    ap.add_argument("--clean-export", type=Path, required=True,
                    help="Dir containing clean fairness_attribution_dense.pkl")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("famail_temporal/results/analysis/sink_decomposition"))
    args = ap.parse_args(argv)

    dirty_pkl = args.dirty_export / "fairness_attribution_dense.pkl"
    clean_pkl = args.clean_export / "fairness_attribution_dense.pkl"
    decompose(dirty_pkl, clean_pkl, DEFAULT_SINK_CELLS, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
