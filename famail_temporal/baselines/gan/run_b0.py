"""CLI: train the B0 generative baseline on the real corpus and report
generated-vs-corpus fairness.

Example:
    python -m famail_temporal.baselines.gan.run_b0 --epochs 5 --device auto
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.b0 import run_b0


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.baselines.gan.run_b0")
    ap.add_argument("--epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results" / "b0")
    args = ap.parse_args(argv)

    bundle = DataBundle.load()
    result = run_b0(
        bundle, epochs=args.epochs, max_len=args.max_len,
        device=_resolve_device(args.device), seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b0_fairness.json").write_text(result_to_json(result))
    print(f"corpus    F_causal={result['corpus']['f_causal']:.4f}")
    print(f"generated F_causal={result['generated']['f_causal']:.4f}")
    print(f"wrote {args.out_dir / 'b0_fairness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
