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

# The 10 calibrated stuck-GPS sink cells (1-indexed, matching the pipeline grid;
# −1 per axis indexes the 0-indexed editor/attribution grid). Single source of
# truth = the source-gen filter's STUCK_GPS_EXPECTED_CELLS. Headline sink (29,53)
# → editor grid [28,52].
from famail_temporal.data.source_generation import config as _sgcfg

DEFAULT_SINK_CELLS: List[Tuple[int, int]] = sorted(
    (int(x), int(y)) for (x, y) in _sgcfg.STUCK_GPS_EXPECTED_CELLS
)


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


def _load_editor_grid_spatial(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an editor run's grid_before.pkl -> (spatial_attr channel, active_mask).

    NO source_data swap needed: the editor already persisted the (gx,gy,T,4)
    fairness grid; channel 0 ('spatial_attr') sums to F_spatial. pickle is safe
    (project-internal numpy artifact from famail's own persistence layer).
    """
    with open(Path(run_dir) / "grid_before.pkl", "rb") as f:
        data = pickle.load(f)
    grid = data["grid"]                       # (gx, gy, T, 4)
    assert data["channel_names"][0] == "spatial_attr", data["channel_names"]
    return grid[..., 0], data["active_mask"]


def decompose_from_editor_grids(
    dirty_run_dir: Path,
    clean_run_dir: Path,
    sink_cells: List[Tuple[int, int]],
    out_dir: Path,
) -> Path:
    """Per-sink F_spatial decomposition straight from two editor runs' grids
    (dirty vs clean), with NO source_data swap. Reuses sink_spatial_contributions.
    Writes ``sink_f_spatial_decomposition.json`` + ``.md`` to out_dir."""
    dirty_spatial, dirty_mask = _load_editor_grid_spatial(dirty_run_dir)
    clean_spatial, clean_mask = _load_editor_grid_spatial(clean_run_dir)
    return _write_decomposition(
        dirty_spatial, dirty_mask, clean_spatial, clean_mask, sink_cells, out_dir,
        provenance={"dirty_run": str(dirty_run_dir), "clean_run": str(clean_run_dir),
                    "source": "editor grid_before.pkl channel-0 (no source_data swap)"},
    )


def _write_decomposition(dirty_spatial, dirty_mask, clean_spatial, clean_mask,
                         sink_cells, out_dir, *, provenance=None) -> Path:
    dirty_contribs = sink_spatial_contributions(dirty_spatial, dirty_mask, sink_cells)
    clean_contribs = sink_spatial_contributions(clean_spatial, clean_mask, sink_cells)
    # Global F_spatial (sum of channel-0 over all active cells), for context.
    dirty_global = float(np.nansum(dirty_spatial))
    clean_global = float(np.nansum(clean_spatial))
    total_shift = clean_global - dirty_global
    per_sink_results: dict = {}
    for cell_key in dirty_contribs["per_sink"]:
        dc = dirty_contribs["per_sink"][cell_key]
        cc = clean_contribs["per_sink"].get(cell_key, 0.0)
        delta = cc - dc
        share = (delta / total_shift) if abs(total_shift) > 1e-12 else None
        per_sink_results[cell_key] = {
            "dirty_contrib": dc, "clean_contrib": cc, "delta": delta,
            "share_of_global_shift": share,
        }
    sinks_delta_sum = float(sum(v["delta"] for v in per_sink_results.values()))
    result = {
        "provenance": provenance or {},
        "global_f_spatial_dirty": dirty_global,
        "global_f_spatial_clean": clean_global,
        "global_shift": total_shift,
        "sinks_delta_sum": sinks_delta_sum,
        "redistribution_residual": total_shift - sinks_delta_sum,
        "per_sink": per_sink_results,
    }
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sink_f_spatial_decomposition.json").write_text(json.dumps(result, indent=2))
    # markdown
    rows = sorted(per_sink_results.items(), key=lambda kv: kv[1]["delta"], reverse=True)
    L = ["# Per-sink F_spatial decomposition (E23): dirty vs clean", "",
         f"Global F_spatial: {dirty_global:.5f} (dirty) -> {clean_global:.5f} (clean), "
         f"shift {total_shift:+.5f}.",
         f"Sum of the {len(per_sink_results)} sink cells' deltas: {sinks_delta_sum:+.5f} "
         f"(redistribution residual on non-sink cells: {total_shift - sinks_delta_sum:+.5f}).", "",
         "| sink (1-idx) | dirty αᵢ | clean αᵢ | delta | share of global shift |",
         "|---|---:|---:|---:|---:|"]
    for k, v in rows:
        sh = f"{v['share_of_global_shift']:.1%}" if v["share_of_global_shift"] is not None else "—"
        L.append(f"| {k} | {v['dirty_contrib']:+.5f} | {v['clean_contrib']:+.5f} | "
                 f"{v['delta']:+.5f} | {sh} |")
    (out_dir / "sink_f_spatial_decomposition.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out_dir/'sink_f_spatial_decomposition.json'}")
    print(f"wrote {out_dir/'sink_f_spatial_decomposition.md'}")
    return out_dir / "sink_f_spatial_decomposition.json"


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
    ap.add_argument("--editor-dirty", type=Path, default=None,
                    help="NO-SWAP mode: dirty editor run dir (uses its grid_before.pkl).")
    ap.add_argument("--editor-clean", type=Path, default=None,
                    help="NO-SWAP mode: clean editor run dir (uses its grid_before.pkl).")
    ap.add_argument("--dirty-export", type=Path, default=None,
                    help="Swap mode: dir containing dirty fairness_attribution_dense.pkl")
    ap.add_argument("--clean-export", type=Path, default=None,
                    help="Swap mode: dir containing clean fairness_attribution_dense.pkl")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("famail_temporal/results/analysis/sink_decomposition"))
    args = ap.parse_args(argv)

    if args.editor_dirty and args.editor_clean:
        decompose_from_editor_grids(args.editor_dirty, args.editor_clean,
                                    DEFAULT_SINK_CELLS, args.out_dir)
    elif args.dirty_export and args.clean_export:
        decompose(args.dirty_export / "fairness_attribution_dense.pkl",
                  args.clean_export / "fairness_attribution_dense.pkl",
                  DEFAULT_SINK_CELLS, args.out_dir)
    else:
        raise SystemExit("provide --editor-dirty/--editor-clean (no-swap) "
                         "OR --dirty-export/--clean-export (swap mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
