"""CLI: model-level metric hardening for the FAMAIL paper headline.

Trains MLE-only B0 (full corpus) and MLE-only FAMAIL (edited corpus), then:
  1. JS terminal-cell transmission check.
  2. Disparate impact (DI) under both Y conventions.
  3. Localized F_causal restricted to the edited active units.

Per-run artifacts: <out-dir>/metrics.json, terminal_cell_histograms.npz,
report.md. The canonical summary at famail_temporal/baselines/metric_hardening/
RESULTS.md is updated by Task 7 once the first real-data run lands.

Example:
    python -m famail_temporal.baselines.run_metric_hardening \\
        --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \\
        --mle-epochs 5 --device auto --seed 0
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.model_level import fit_and_evaluate
from famail_temporal.baselines.gan.variants import load_edited_trajectories
from famail_temporal.baselines.gan.rollout import pickups_to_pickup_3d
from famail_temporal.baselines.transmission import (
    terminal_cell_histogram, trajectories_terminal_histogram, transmission_metrics,
)
from famail_temporal.baselines.district_metrics import di_from_bundle_and_pickup_grid
from famail_temporal.baselines.localized_metrics import (
    edited_units_from_histories, localized_f_causal,
)


DEFAULT_EDIT_DIR = (
    Path(config.PACKAGE_ROOT) / "results"
    / "2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup"
)


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=float)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _write_report(out_dir: Path, result: dict, cmd: List[str]) -> None:
    """Per-run human-readable report (markdown). Paper-ready prose."""
    t = result["transmission"]
    di_b0 = result["di_b0"]
    di_fam = result["di_famail"]
    loc_b0 = result["localized_b0"]
    loc_fam = result["localized_famail"]
    report = f"""# Metric hardening run report

**Command**: `{' '.join(cmd)}`
**Edit source**: `{result['edit_dir']}`

## Transmission (does the data-level signal survive the LSTM?)

| Quantity | Value |
|---|---|
| JS(p_raw, p_edited) - *target* shift | **{t['js_target']:.5f}** bits |
| JS(p_gen_B0, p_gen_FAMAIL) - *transmitted* shift | **{t['js_generated']:.5f}** bits |
| **Transmission ratio** (transmitted / target) | **{t['transmission_ratio']:.3f}** |
| JS(p_gen_B0, p_raw) - B0 fidelity to raw target | {t['js_b0_vs_raw']:.5f} |
| JS(p_gen_FAMAIL, p_edited) - FAMAIL fidelity to edited target | {t['js_famail_vs_edited']:.5f} |

Reading: transmission_ratio ~ 1.0 means the generator faithfully transmitted
the edit; << 1 means MLE smoothing + multinomial sampling washed it out.

## Disparate impact (DI) - both Y conventions

|       | Y = supply/demand (primary; F_causal-aligned) | Y = demand/supply (supplementary) |
|---|---:|---:|
| B0     | {di_b0['di_primary']:.4f} | {di_b0['di_supplementary']:.4f} |
| FAMAIL | {di_fam['di_primary']:.4f} | {di_fam['di_supplementary']:.4f} |
| Delta DI | **{di_fam['di_primary'] - di_b0['di_primary']:+.4f}** | **{di_fam['di_supplementary'] - di_b0['di_supplementary']:+.4f}** |

Top-3 hukou districts: {di_b0.get('top_district_ids', [])}; bottom-3: {di_b0.get('bottom_district_ids', [])}.
Both DIs should move in the *same* direction under FAMAIL editing (robustness).

## Localized F_causal (restricted to {loc_b0['n_edited_active_units']} edited active units)

|       | F_causal_global | F_causal_localized |
|---|---:|---:|
| B0     | {loc_b0['f_causal_global']:.4f} | {loc_b0['f_causal_localized']:.4f} |
| FAMAIL | {loc_fam['f_causal_global']:.4f} | {loc_fam['f_causal_localized']:.4f} |
| Delta  | {loc_fam['f_causal_global'] - loc_b0['f_causal_global']:+.4f} | **{loc_fam['f_causal_localized'] - loc_b0['f_causal_localized']:+.4f}** |

Note: f_causal_global here uses M=I (uniform weighting), the same formula as
f_causal_localized at different N. This is NOT the production F_causal in
b0_fairness/famail_fairness (which uses M=center). See MODEL_LEVEL_METRICS.md.

Reading: localized Delta should be substantially larger than global Delta because
the edit's effect concentrates in the touched units. If localized Delta is also
small, the headline is fragile and the data-level Pareto is the more honest framing.
"""
    (out_dir / "report.md").write_text(report)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_metric_hardening",
    )
    ap.add_argument("--mle-epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--mle-batch-size", type=int, default=gc.MLE_BATCH_SIZE)
    ap.add_argument("--max-tokens", type=int, default=gc.MAX_TRAIN_TOKENS)
    ap.add_argument("--gen-batch-size", type=int, default=gc.GEN_BATCH_SIZE)
    ap.add_argument("--edit-dir", type=Path, default=DEFAULT_EDIT_DIR,
                    help="Results dir with histories.pkl for the FAMAIL edit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    bundle = DataBundle.load()
    edited_trajs = load_edited_trajectories(bundle, args.edit_dir)

    # --- Train B0 (MLE-only, full corpus) ---
    b0 = fit_and_evaluate(
        bundle,
        mle_epochs=args.mle_epochs, adv_epochs=0, max_len=args.max_len,
        mle_batch_size=args.mle_batch_size,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        gen_batch_size=args.gen_batch_size,
        device=device, seed=args.seed, progress=not args.quiet,
    )

    # --- Train FAMAIL (MLE-only, edited corpus) ---
    famail = fit_and_evaluate(
        bundle, train_trajectories=edited_trajs,
        mle_epochs=args.mle_epochs, adv_epochs=0, max_len=args.max_len,
        mle_batch_size=args.mle_batch_size,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        gen_batch_size=args.gen_batch_size,
        device=device, seed=args.seed, progress=not args.quiet,
    )

    # --- Build histograms (terminal cells only - what the metric sees) ---
    p_raw = trajectories_terminal_histogram(bundle.trajectories)
    p_edited = trajectories_terminal_histogram(edited_trajs)
    p_gen_b0 = terminal_cell_histogram(b0["pickups"])
    p_gen_famail = terminal_cell_histogram(famail["pickups"])

    transmission = transmission_metrics(p_raw, p_edited, p_gen_b0, p_gen_famail)

    # --- DI on generated grids (uses each variant's pickup grid) ---
    b0_grid = pickups_to_pickup_3d(bundle, b0["pickups"])
    famail_grid = pickups_to_pickup_3d(bundle, famail["pickups"])
    di_b0 = di_from_bundle_and_pickup_grid(bundle, b0_grid)
    di_famail = di_from_bundle_and_pickup_grid(bundle, famail_grid)

    # --- Localized F_causal on the same grids, restricted to edited units ---
    edited_units = edited_units_from_histories(args.edit_dir)
    loc_b0 = localized_f_causal(bundle, b0_grid, edited_units)
    loc_famail = localized_f_causal(bundle, famail_grid, edited_units)

    result = {
        "transmission": transmission,
        "di_b0": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                  for k, v in di_b0.items()},
        "di_famail": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                      for k, v in di_famail.items()},
        "localized_b0": loc_b0,
        "localized_famail": loc_famail,
        "edit_dir": str(args.edit_dir),
        "b0_fairness": b0["generated"],
        "famail_fairness": famail["generated"],
        "corpus_fairness": b0["corpus"],
        "n_generated": b0["n_generated"],
        "mle_losses_b0": b0["mle_losses"],
        "mle_losses_famail": famail["mle_losses"],
    }

    # --- Persist ---
    if args.out_dir is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = (
            Path(config.PACKAGE_ROOT) / "baselines" / "metric_hardening"
            / "results" / f"{timestamp}_metric_hardening"
        )
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(result_to_json(result))
    np.savez(
        out_dir / "terminal_cell_histograms.npz",
        p_raw=p_raw, p_edited=p_edited,
        p_gen_b0=p_gen_b0, p_gen_famail=p_gen_famail,
    )
    cmd = ["python", "-m", "famail_temporal.baselines.run_metric_hardening"]
    if argv:
        cmd += argv
    _write_report(out_dir, result, cmd)

    print(f"\n=== Metric hardening summary ===")
    print(f"Transmission ratio: {transmission['transmission_ratio']:.3f} "
          f"(JS_target={transmission['js_target']:.5f}, "
          f"JS_generated={transmission['js_generated']:.5f})")
    print(f"DI_primary       B0={di_b0['di_primary']:.4f}  "
          f"FAMAIL={di_famail['di_primary']:.4f}  "
          f"DeltaDI={di_famail['di_primary'] - di_b0['di_primary']:+.4f}")
    print(f"F_causal local   B0={loc_b0['f_causal_localized']:.4f}  "
          f"FAMAIL={loc_famail['f_causal_localized']:.4f}  "
          f"Delta={loc_famail['f_causal_localized'] - loc_b0['f_causal_localized']:+.4f}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
