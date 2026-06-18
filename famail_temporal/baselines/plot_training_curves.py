"""CLI utility: read persisted training-curve data and export CSV + PNG.

Supports two input modes:
  --level1-dir DIR   Read training_curves.json (bc + gan curves).
  --variance-dir DIR  Read seed_*.json files (b0 + famail per-seed curves).
  --out-dir DIR       Destination directory (default: <input dir>/curves).

Headless: uses the Agg backend — safe in environments without a display.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def series_csv(values: list[float]) -> str:
    """CSV text 'step,loss\\n0,<v0>\\n1,<v1>\\n...' for a 1-D loss series."""
    lines = ["step,loss"]
    for i, v in enumerate(values):
        lines.append(f"{i},{v}")
    return "\n".join(lines) + "\n"


def flatten_level1_curves(curves: dict) -> dict[str, list]:
    """Level-1 training_curves dict -> {series_name: values}.

    Series names (omit any whose source list is empty/absent):
      bc_mle_epoch, bc_mle_batch,
      gan_mle_epoch, gan_mle_batch,
      gan_adv_g_epoch, gan_adv_d_epoch, gan_adv_g_batch, gan_adv_d_batch
    (BC has adv=None -> no bc_adv_* series.)
    """
    result: dict[str, list] = {}

    def _add(name: str, values: list | None) -> None:
        if values:  # non-None and non-empty
            result[name] = values

    # BC
    bc = curves.get("bc", {})
    _add("bc_mle_epoch", bc.get("mle_epoch_losses"))
    _add("bc_mle_batch", bc.get("mle_batch_losses"))
    # BC adv is always None per spec — no bc_adv_* keys

    # GAN MLE
    gan = curves.get("gan", {})
    _add("gan_mle_epoch", gan.get("mle_epoch_losses"))
    _add("gan_mle_batch", gan.get("mle_batch_losses"))

    # GAN adversarial
    adv = gan.get("adv") or {}
    if adv:
        _add("gan_adv_g_epoch", adv.get("g_epoch_losses"))
        _add("gan_adv_d_epoch", adv.get("d_epoch_losses"))
        _add("gan_adv_g_batch", adv.get("g_batch_losses"))
        _add("gan_adv_d_batch", adv.get("d_batch_losses"))

    return result


def variance_model_series(seed_entries: list[dict], model: str) -> dict[str, list]:
    """Per-seed MLE curves for 'b0' or 'famail' across seed entries.

    Returns {f"{model}_seed{seed}_mle": <curve>} where <curve> is the seed's
    mle_batch_losses if present and non-empty, else its mle_losses (per-epoch
    fallback for OLD files). Skip a seed if neither is present.
    """
    result: dict[str, list] = {}
    for entry in seed_entries:
        seed_id = entry["seed"]
        model_data = entry.get(model, {})

        batch = model_data.get("mle_batch_losses")
        epoch = model_data.get("mle_losses")

        if batch:  # non-None and non-empty
            curve = batch
        elif epoch:  # fallback for old files
            curve = epoch
        else:
            continue  # skip — no usable curve

        result[f"{model}_seed{seed_id}_mle"] = curve
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def plot_series_group(
    title: str, series_map: dict[str, list], out_png: Path
) -> Path:
    """Plot each named series (y=loss vs x=index) on one figure; save PNG.

    Returns out_png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in series_map.items():
        ax.plot(range(len(values)), values, label=name)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    if series_map:
        ax.legend(fontsize="small")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------


def _write_csv(out_dir: Path, name: str, values: list[float]) -> Path:
    csv_path = out_dir / f"{name}.csv"
    csv_path.write_text(series_csv(values))
    return csv_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export training curves as CSV + PNG."
    )
    parser.add_argument(
        "--level1-dir",
        metavar="DIR",
        help="Run directory containing training_curves.json",
    )
    parser.add_argument(
        "--variance-dir",
        metavar="DIR",
        help="Run directory containing seed_*.json files",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help="Output directory (default: <input dir>/curves)",
    )
    args = parser.parse_args(argv)

    if not args.level1_dir and not args.variance_dir:
        parser.error("At least one of --level1-dir or --variance-dir is required.")

    written: list[Path] = []

    # -----------------------------------------------------------------------
    # Level-1 mode
    # -----------------------------------------------------------------------
    if args.level1_dir:
        level1_dir = Path(args.level1_dir)
        out_dir = Path(args.out_dir) if args.out_dir else level1_dir / "curves"
        out_dir.mkdir(parents=True, exist_ok=True)

        curves_path = level1_dir / "training_curves.json"
        with curves_path.open() as f:
            curves = json.load(f)

        flat = flatten_level1_curves(curves)

        # Write one CSV per series
        for name, values in flat.items():
            written.append(_write_csv(out_dir, name, values))

        # --- PNG groups ---

        # BC MLE: prefer batch, else epoch
        bc_mle_series: dict[str, list] = {}
        if "bc_mle_batch" in flat:
            bc_mle_series["bc_mle_batch"] = flat["bc_mle_batch"]
        elif "bc_mle_epoch" in flat:
            bc_mle_series["bc_mle_epoch"] = flat["bc_mle_epoch"]
        if bc_mle_series:
            written.append(
                plot_series_group("BC MLE Loss", bc_mle_series, out_dir / "bc_mle.png")
            )

        # GAN MLE: prefer batch, else epoch
        gan_mle_series: dict[str, list] = {}
        if "gan_mle_batch" in flat:
            gan_mle_series["gan_mle_batch"] = flat["gan_mle_batch"]
        elif "gan_mle_epoch" in flat:
            gan_mle_series["gan_mle_epoch"] = flat["gan_mle_epoch"]
        if gan_mle_series:
            written.append(
                plot_series_group(
                    "GAN MLE Loss", gan_mle_series, out_dir / "gan_mle.png"
                )
            )

        # GAN adversarial: prefer batch curves, else epoch
        adv_keys_batch = ["gan_adv_g_batch", "gan_adv_d_batch"]
        adv_keys_epoch = ["gan_adv_g_epoch", "gan_adv_d_epoch"]
        adv_series: dict[str, list] = {}
        for k in adv_keys_batch:
            if k in flat:
                adv_series[k] = flat[k]
        if not adv_series:
            for k in adv_keys_epoch:
                if k in flat:
                    adv_series[k] = flat[k]
        if adv_series:
            written.append(
                plot_series_group(
                    "GAN Adversarial Losses",
                    adv_series,
                    out_dir / "gan_adversarial.png",
                )
            )

    # -----------------------------------------------------------------------
    # Variance-suite mode
    # -----------------------------------------------------------------------
    if args.variance_dir:
        var_dir = Path(args.variance_dir)
        out_dir = Path(args.out_dir) if args.out_dir else var_dir / "curves"
        out_dir.mkdir(parents=True, exist_ok=True)

        seed_files = sorted(var_dir.glob("seed_*.json"))
        seed_entries: list[dict] = []
        for sf in seed_files:
            with sf.open() as f:
                seed_entries.append(json.load(f))

        for model in ("b0", "famail"):
            model_series = variance_model_series(seed_entries, model)

            # Write one CSV per seed curve
            for name, values in model_series.items():
                written.append(_write_csv(out_dir, name, values))

            # Write one overlaid PNG per model
            if model_series:
                written.append(
                    plot_series_group(
                        f"{model.upper()} MLE — all seeds",
                        model_series,
                        out_dir / f"{model}_mle.png",
                    )
                )

    for p in written:
        print(p)

    return 0


if __name__ == "__main__":
    sys.exit(main())
