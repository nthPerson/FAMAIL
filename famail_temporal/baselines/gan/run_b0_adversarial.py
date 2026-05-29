"""CLI: train the standard-adversarial B0 (MLE + Gumbel adversarial fine-tune)
on the real corpus and report generated-vs-corpus fairness.

Example:
    python -m famail_temporal.baselines.gan.run_b0_adversarial \
        --mle-epochs 5 --adv-epochs 3 --device auto
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
from famail_temporal.baselines.gan.model_level import fit_and_evaluate


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.gan.run_b0_adversarial",
    )
    ap.add_argument("--mle-epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--adv-epochs", type=int, default=gc.ADV_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--mle-batch-size", type=int, default=gc.MLE_BATCH_SIZE)
    ap.add_argument("--adv-batch-size", type=int, default=gc.ADV_BATCH_SIZE)
    ap.add_argument("--adv-lr-g", type=float, default=gc.ADV_LR_G)
    ap.add_argument("--adv-lr-d", type=float, default=gc.ADV_LR_D,
                    help="critic LR; lower it to slow a dominating critic")
    ap.add_argument("--d-update-every", type=int, default=gc.D_UPDATE_EVERY,
                    help="update the critic every k-th batch; raise to slow it")
    ap.add_argument("--adv-mle-lambda", type=float, default=gc.ADV_MLE_LAMBDA,
                    help="weight on the teacher-forced MLE anchor in the "
                         "generator loss (0 disables; prevents drift/collapse)")
    ap.add_argument("--adv-max-len", type=int, default=None,
                    help="opt-in cap on the adversarial rollout length (tokens) "
                         "as a hard backstop against fake-length blowup; "
                         "defaults to --max-len")
    ap.add_argument("--gen-batch-size", type=int, default=gc.GEN_BATCH_SIZE,
                    help="contexts decoded in parallel during generation")
    ap.add_argument("--max-tokens", type=int, default=gc.MAX_TRAIN_TOKENS,
                    help="exclude trajectories longer than this from training "
                         "(<=0 disables the filter)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true",
                    help="suppress progress bars / phase markers")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results" / "b0_adversarial")
    args = ap.parse_args(argv)

    bundle = DataBundle.load()
    result = fit_and_evaluate(
        bundle, mle_epochs=args.mle_epochs, adv_epochs=args.adv_epochs,
        max_len=args.max_len, mle_batch_size=args.mle_batch_size,
        adv_batch_size=args.adv_batch_size,
        adv_lr_g=args.adv_lr_g, adv_lr_d=args.adv_lr_d,
        d_update_every=args.d_update_every, adv_mle_lambda=args.adv_mle_lambda,
        adv_max_len=args.adv_max_len, gen_batch_size=args.gen_batch_size,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        device=_resolve_device(args.device), seed=args.seed,
        progress=not args.quiet,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b0_adversarial_fairness.json").write_text(
        result_to_json(result)
    )
    print(f"corpus    F_causal={result['corpus']['f_causal']:.4f}")
    print(f"generated F_causal={result['generated']['f_causal']:.4f}")
    print(f"wrote {args.out_dir / 'b0_adversarial_fairness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
