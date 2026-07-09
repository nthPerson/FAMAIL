"""Supply-vs-demand channel decomposition of the supply-lift Δmean(Y|D)
(Task 11a; paper-load-bearing).

The supply-lift edit changes mean service ratio ``Y = supply / demand`` on the
disadvantaged group ``D`` (migrant-axis district extremes) through TWO
channels:

* **demand** — relocating pickups changes each cell's demand ``D``;
* **supply** — the endogenous tier-1 ΔS changes each cell's supply ``S``.

We decompose the total ``Δmean(Y|D)`` sequentially. With
``Y(S, D) = S / max(D, DEMAND_FLOOR)`` evaluated per active unit and averaged
over the disadvantaged group:

    total  = mean_D[ Y(S', D') - Y(S_base, D_base) ]
    demand = mean_D[ Y(S_base, D') - Y(S_base, D_base) ]   (supply held at base)
    supply = mean_D[ Y(S', D')     - Y(S_base, D') ]        (demand held at D')

so ``total = demand + supply`` (demand-first ordering). Two robustness lines:

* **supply-first** ordering:
    supply_first  = mean_D[ Y(S', D_base) - Y(S_base, D_base) ]
    demand_second = mean_D[ Y(S', D')     - Y(S', D_base) ]
* **tier-2 supply channel** — substitute the distinct-count AFTER-supply grid
  ``S_tier2_after`` (from ``supply_recount --persist-grids``) for ``S'`` in the
  supply channel (demand channel unchanged):
    supply_tier2 = mean_D[ Y(S_tier2, D') - Y(S_base, D') ]
    total_tier2  = mean_D[ Y(S_tier2, D') - Y(S_base, D_base) ]

CIs: paired bootstrap over the disadvantaged-group units (shared resample per
replicate), ``B`` resamples (default 2000), seed 0, 95% percentile CIs.

CLI::

    python -m famail_temporal.analysis.channel_decomposition \\
        --edit-dir <results_dir> [--tier2-grid <S_tier2_after.npz>] \\
        [--group-axis MigrantRatio] [--bootstrap 2000] [--seed 0]

Read-only; does not modify any existing module or the edit dir (unless
``--out`` names a path inside it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def _mean_or_nan(v: np.ndarray) -> float:
    return float(v.mean()) if v.size else float("nan")


def bootstrap_channels(
    channel_vectors: Dict[str, np.ndarray],
    B: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, dict]:
    """Paired bootstrap over group units. ``channel_vectors[name]`` is a
    per-unit vector (all the same length N_D); each replicate draws one shared
    index set and averages every channel on it. Returns point estimate + CI
    per channel."""
    names = list(channel_vectors)
    N = len(channel_vectors[names[0]]) if names else 0
    for name, v in channel_vectors.items():
        if len(v) != N:
            raise ValueError(f"channel {name!r} length {len(v)} != {N}")
    lo_q, hi_q = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0
    rng = np.random.default_rng(seed)
    reps = {name: np.empty(B) for name in names}
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        for name, v in channel_vectors.items():
            reps[name][b] = v[idx].mean()
    out: Dict[str, dict] = {}
    for name, v in channel_vectors.items():
        r = reps[name]
        out[name] = {
            "point": _mean_or_nan(v),
            "ci_lo": float(np.percentile(r, lo_q)),
            "ci_hi": float(np.percentile(r, hi_q)),
            "significant": bool(np.percentile(r, lo_q) > 0 or np.percentile(r, hi_q) < 0),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edit-dir", type=Path, required=True,
                    help="Results dir with histories.pkl + delta_supply_3d.npz.")
    ap.add_argument("--tier2-grid", type=Path, default=None,
                    help="Path to S_tier2_after.npz (from supply_recount "
                         "--persist-grids); enables the tier-2 supply channel.")
    ap.add_argument("--group-axis", default="MigrantRatio",
                    help="Equity axis defining the disadvantaged group D.")
    ap.add_argument("--disadvantaged-high", type=lambda s: s.lower() != "false",
                    default=True, help="True => high axis value is disadvantaged "
                                       "(migrant convention).")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output path (default: <edit_dir>/channel_decomposition.json).")
    args = ap.parse_args(argv)

    from famail_temporal import config
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.baselines import external_fairness as ef
    from famail_temporal.baselines import external_fairness_io as efio

    edit_dir = Path(args.edit_dir)
    out_path = Path(args.out) if args.out else edit_dir / "channel_decomposition.json"

    print("[channel] loading bundle...", flush=True)
    bundle = DataBundle.load()

    S_base = bundle.active_taxis_3d
    D_base = bundle.pickup_3d

    delta = np.load(edit_dir / "delta_supply_3d.npz")["delta_supply_3d"]
    S_prime = np.clip(S_base + delta, config.SUPPLY_FLOOR, None).astype(S_base.dtype)

    # Edited demand (same clip/sanitize convention as runner.py).
    D_prime = np.clip(efio.build_edited_pickup_3d(bundle, edit_dir), 0.0, None)

    # Per-unit Y vectors for each (supply, demand) combination.
    from dataclasses import replace
    b_base = bundle
    b_prime = replace(bundle, active_taxis_3d=S_prime)
    Y_bb = efio.service_ratio_Y(D_base, b_base)                 # Y(S_base, D_base)
    Y_bp = efio.service_ratio_Y(D_prime, b_base)                # Y(S_base, D')
    Y_pp = efio.service_ratio_Y(D_prime, b_prime)              # Y(S', D')
    Y_pb = efio.service_ratio_Y(D_base, b_prime)               # Y(S', D_base)

    # Disadvantaged group D (migrant-axis district extremes).
    demo = efio.per_unit_demographics(bundle)
    g_unit = ef.region_extremes(demo[args.group_axis],
                                disadvantaged_high=args.disadvantaged_high)
    d = g_unit == 1
    N_D = int(d.sum())
    print(f"[channel] group axis={args.group_axis} disadvantaged_high="
          f"{args.disadvantaged_high} -> N_D={N_D} units", flush=True)

    yb, ybp, ypp, ypb = Y_bb[d], Y_bp[d], Y_pp[d], Y_pb[d]

    channels: Dict[str, np.ndarray] = {
        # demand-first sequential decomposition
        "demand": ybp - yb,
        "supply": ypp - ybp,
        "total": ypp - yb,
        # supply-first robustness
        "supply_first": ypb - yb,
        "demand_second": ypp - ypb,
    }
    levels = {
        "mean_Y_D_before": _mean_or_nan(yb),
        "mean_Y_D_after_tier1": _mean_or_nan(ypp),
    }

    tier2_meta = None
    if args.tier2_grid is not None:
        S_t2 = np.load(args.tier2_grid)["S_tier2_after"].astype(S_base.dtype)
        b_t2 = replace(bundle, active_taxis_3d=S_t2)
        Y_t2 = efio.service_ratio_Y(D_prime, b_t2)             # Y(S_tier2, D')
        yt2 = Y_t2[d]
        channels["supply_tier2"] = yt2 - ybp                   # demand channel unchanged
        channels["total_tier2"] = yt2 - yb
        levels["mean_Y_D_after_tier2"] = _mean_or_nan(yt2)
        tier2_meta = str(args.tier2_grid)

    print(f"[channel] bootstrapping B={args.bootstrap} seed={args.seed}...", flush=True)
    boot = bootstrap_channels(channels, B=args.bootstrap, seed=args.seed)

    result = {
        "edit_dir": str(edit_dir),
        "group_axis": args.group_axis,
        "disadvantaged_high": bool(args.disadvantaged_high),
        "N_D": N_D,
        "bootstrap": {"B": args.bootstrap, "seed": args.seed, "ci": 0.95},
        "tier2_grid": tier2_meta,
        "levels": levels,
        "channels": boot,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[channel] wrote {out_path}", flush=True)

    # Console summary.
    def _fmt(name):
        c = boot[name]
        star = "  *SIGNIFICANT*" if c["significant"] else "  (spans 0)"
        return f"  {name:16s} {c['point']:+.6f}  CI[{c['ci_lo']:+.6f}, {c['ci_hi']:+.6f}]{star}"
    print("[channel] demand-first decomposition (mean Y | migrant D-group):", flush=True)
    for n in ("total", "demand", "supply"):
        print(_fmt(n), flush=True)
    print("[channel] supply-first robustness:", flush=True)
    for n in ("supply_first", "demand_second"):
        print(_fmt(n), flush=True)
    if tier2_meta is not None:
        print("[channel] tier-2 (distinct-count) supply channel:", flush=True)
        for n in ("supply_tier2", "total_tier2"):
            print(_fmt(n), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
