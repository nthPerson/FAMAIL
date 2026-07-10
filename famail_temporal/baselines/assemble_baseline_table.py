"""Mission-3 Task 5: 5-row baseline comparison-table assembler.

Pure JSON-in -> markdown/json-out. No torch import, no recomputation of any
headline number — every value in the output table is read verbatim (or
computed as a simple after-before delta) from inputs already on disk:

- Arm dirs (Tasks 2-4 output): each ``<arm-dir>/metrics.json`` with the
  schema written by ``stifgsm_baseline.package_arm`` +
  ``run_stifgsm_baseline._rescore`` / ``score_fidelity``::

      {
        "arm": {
          "mode": "ifgsm" | "fgsm" | "random", "epsilon": ..., "seed": ...,
          "n_edited": int, "adjacency_violation_rate": float (0-1 fraction),
          "mean_final_p": float, "mean_iterations": float, ...
        },
        "fairness": {
          "f_spatial_before": float, "f_spatial_after": float,
          "f_causal_before": float, "f_causal_after": float,
          "deltas": {"f_spatial": float, "f_causal": float}   # optional
        },
        "fidelity": {                                          # OPTIONAL block
          "fidelity_a": {"mean": float, "std": float, "n": int, ...},
          "gate": {"passed": bool, ...},
          "fidelity_b": {"per_stat": {...}, "terminal_cell_js": float,
                         "aggregate": float}
        }
      }

  The row label is the arm's own ``arm.mode`` (not the CLI position).

- ``--famail-json`` / ``--raw-json``: small HAND-AUTHORED stub files (the
  controller/user transcribes the published headline numbers into these; we
  never recompute them). Same column semantics as the arm schema above,
  flattened::

      {
        "label": "FAMAIL",                  # optional; defaults per-arg
        "fidelity_a": float | null,          # Fidelity-A mean
        "gate_passed": bool | null,
        "fidelity_b": float | null,          # Fidelity-B aggregate JS
        "f_causal_before": float | null,
        "f_causal_after": float | null,
        "f_spatial_before": float | null,
        "f_spatial_after": float | null,
        "adjacency_violation_rate": float | null,   # 0-1 fraction
        "mean_final_p": float | null,
        "n": int | null
      }

  Any field may be null/absent -> renders as "—" in the markdown table
  (never a KeyError). The raw row's deltas are 0/None by definition: supply
  equal (or null) before/after so the after-before computation yields 0 (or
  None), same formula used for every other row — no special-casing.

Output columns (both `baseline_table.md` and `baseline_table.json`):
    [Fidelity-A, gate, Fidelity-B(JS), ΔF_causal, ΔF_spatial,
     adjacency-violation %, mean final_p, n]

Row order: raw, FAMAIL, then the arm dirs in the order given on the CLI.

The JSON output keeps full float precision; the markdown table rounds
floats to 4 decimals (adjacency shown as a percent with 1 decimal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keys of the flattened per-row representation shared by arm dirs and the
# hand-authored famail/raw JSON stubs.
_FLAT_KEYS = (
    "fidelity_a",
    "gate_passed",
    "fidelity_b",
    "f_causal_before",
    "f_causal_after",
    "f_spatial_before",
    "f_spatial_after",
    "adjacency_violation_rate",
    "mean_final_p",
    "n",
)

_COLUMNS = (
    "Fidelity-A",
    "gate",
    "Fidelity-B(JS)",
    "ΔF_causal",
    "ΔF_spatial",
    "adjacency-violation %",
    "mean final_p",
    "n",
)

_EM_DASH = "—"


def _flatten_arm_metrics(meta: Dict[str, Any]) -> Dict[str, Any]:
    """arm-dir metrics.json (Tasks 2-4 schema) -> the shared flat dict."""
    arm = meta.get("arm", {}) or {}
    fairness = meta.get("fairness", {}) or {}
    fidelity = meta.get("fidelity") or {}
    fidelity_a = fidelity.get("fidelity_a") or {}
    gate = fidelity.get("gate") or {}
    fidelity_b = fidelity.get("fidelity_b") or {}
    return {
        "label": arm.get("mode"),
        "fidelity_a": fidelity_a.get("mean"),
        "gate_passed": gate.get("passed"),
        "fidelity_b": fidelity_b.get("aggregate"),
        "f_causal_before": fairness.get("f_causal_before"),
        "f_causal_after": fairness.get("f_causal_after"),
        "f_spatial_before": fairness.get("f_spatial_before"),
        "f_spatial_after": fairness.get("f_spatial_after"),
        "adjacency_violation_rate": arm.get("adjacency_violation_rate"),
        "mean_final_p": arm.get("mean_final_p"),
        "n": arm.get("n_edited"),
    }


def _flatten_stub(data: Dict[str, Any], default_label: str) -> Dict[str, Any]:
    """famail/raw hand-authored JSON stub -> the shared flat dict.

    Missing keys default to None (renders as an em-dash downstream) — never
    a KeyError.
    """
    flat = {"label": data.get("label", default_label)}
    for key in _FLAT_KEYS:
        flat[key] = data.get(key)
    return flat


def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before is None:
        return None
    return after - before


def _build_row(flat: Dict[str, Any]) -> Dict[str, Any]:
    """flat per-source dict -> the output row (adds computed deltas/pct)."""
    adj_rate = flat.get("adjacency_violation_rate")
    adj_pct = None if adj_rate is None else adj_rate * 100.0
    return {
        "label": flat["label"],
        "fidelity_a": flat.get("fidelity_a"),
        "gate_passed": flat.get("gate_passed"),
        "fidelity_b": flat.get("fidelity_b"),
        "delta_f_causal": _delta(flat.get("f_causal_after"), flat.get("f_causal_before")),
        "delta_f_spatial": _delta(flat.get("f_spatial_after"), flat.get("f_spatial_before")),
        "adjacency_violation_pct": adj_pct,
        "mean_final_p": flat.get("mean_final_p"),
        "n": flat.get("n"),
    }


def build_rows(arm_dirs: List[Path], famail_json: Path, raw_json: Path) -> List[Dict[str, Any]]:
    """Assemble the 5(+) ordered rows: raw, FAMAIL, then each arm dir in order."""
    raw_flat = _flatten_stub(json.loads(Path(raw_json).read_text()), default_label="raw")
    famail_flat = _flatten_stub(json.loads(Path(famail_json).read_text()), default_label="FAMAIL")

    rows = [_build_row(raw_flat), _build_row(famail_flat)]
    for arm_dir in arm_dirs:
        meta_path = Path(arm_dir) / "metrics.json"
        meta = json.loads(meta_path.read_text())
        rows.append(_build_row(_flatten_arm_metrics(meta)))
    return rows


# ------------------------------------------------------------------ render ----

def _fmt_float_md(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return _EM_DASH
    return f"{value:.{decimals}f}"


def _fmt_gate_md(passed: Optional[bool]) -> str:
    if passed is None:
        return _EM_DASH
    return "PASS" if passed else "FAIL"


def _fmt_pct_md(value: Optional[float]) -> str:
    if value is None:
        return _EM_DASH
    return f"{value:.1f}%"


def _fmt_n_md(value) -> str:
    if value is None:
        return _EM_DASH
    return str(int(value))


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    header = "| row | " + " | ".join(_COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 1)
    lines = [header, sep]
    for row in rows:
        cells = [
            row["label"],
            _fmt_float_md(row["fidelity_a"]),
            _fmt_gate_md(row["gate_passed"]),
            _fmt_float_md(row["fidelity_b"]),
            _fmt_float_md(row["delta_f_causal"]),
            _fmt_float_md(row["delta_f_spatial"]),
            _fmt_pct_md(row["adjacency_violation_pct"]),
            _fmt_float_md(row["mean_final_p"]),
            _fmt_n_md(row["n"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_json(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"columns": ["row", *_COLUMNS], "rows": rows}


# ------------------------------------------------------------------ cli -------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Assemble the Mission-3 5-row baseline comparison table "
                    "(raw, FAMAIL, ifgsm, fgsm, random) from arm metrics.json "
                    "files + hand-authored FAMAIL/raw headline stubs.")
    p.add_argument("--arm-dirs", nargs="+", required=True,
                   help="One or more arm output dirs, each containing metrics.json; "
                        "rows are labeled by each arm's own arm.mode, in this order.")
    p.add_argument("--famail-json", required=True,
                   help="Path to the hand-authored FAMAIL headline stub (see module docstring).")
    p.add_argument("--raw-json", required=True,
                   help="Path to the hand-authored raw-baseline headline stub (see module docstring).")
    p.add_argument("--out", required=True,
                   help="Output dir; writes baseline_table.md + baseline_table.json.")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    rows = build_rows(
        arm_dirs=[Path(d) for d in args.arm_dirs],
        famail_json=Path(args.famail_json),
        raw_json=Path(args.raw_json),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_table.md").write_text(render_markdown(rows))
    (out_dir / "baseline_table.json").write_text(json.dumps(render_json(rows), indent=2))
    return out_dir


if __name__ == "__main__":
    main()
