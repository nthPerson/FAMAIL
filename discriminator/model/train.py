#!/usr/bin/env python
"""Command-line training script for the discriminator model.

Usage:
    # Train V2 (single-stream, seeking only)
    python -m discriminator.model.train --data-dir ./datasets/single_stream/

    # Train V3 (multi-stream, Ren-aligned)
    python -m discriminator.model.train --model-version v3 \
        --data-dir discriminator/multi_stream/datasets/default/ \
        --combination-mode concatenation --lr 6e-5

    # From the model/ directory
    python train.py --data dataset.npz --model-version v2
"""

import argparse
import json
from pathlib import Path
import sys

import torch

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from model import (
        SiameseLSTMDiscriminator,
        SiameseLSTMDiscriminatorV2,
        MultiStreamSiameseDiscriminator,
        load_dataset_from_directory,
        Trainer,
        TrainingConfig,
    )
    from model.dataset import (
        TrajectoryPairDataset,
        create_train_val_split,
        create_data_loaders,
    )
except ImportError:
    from model import (
        SiameseLSTMDiscriminator,
        SiameseLSTMDiscriminatorV2,
        MultiStreamSiameseDiscriminator,
    )
    from dataset import (
        TrajectoryPairDataset,
        load_dataset_from_directory,
        create_train_val_split,
        create_data_loaders,
    )
    from trainer import Trainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the discriminator model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data", type=str,
                            help="Path to single .npz dataset file (will auto-split)")
    data_group.add_argument("--data-dir", type=str,
                            help="Path to directory with train.npz, val.npz")
    data_group.add_argument("--train", type=str,
                            help="Path to training .npz file")
    data_group.add_argument("--val", type=str,
                            help="Path to validation .npz file")
    data_group.add_argument("--val-split", type=float, default=0.2,
                            help="Validation split ratio when using --data (default: 0.2)")

    # Model architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--model-version", type=str, default="v2",
                             choices=["v1", "v2", "v3"],
                             help="Model version (default: v2)")
    model_group.add_argument("--lstm-hidden-dims", type=str, default="200,100",
                             help="LSTM hidden dims per layer (comma-separated, default: 200,100)")
    model_group.add_argument("--dropout", type=float, default=0.2,
                             help="Dropout probability (default: 0.2)")
    model_group.add_argument("--no-bidirectional", action="store_true",
                             help="Use unidirectional LSTM")
    model_group.add_argument("--classifier-dims", type=str, default="64,32,8",
                             help="Classifier hidden dims (comma-separated, default: 64,32,8)")
    model_group.add_argument("--combination-mode", type=str, default="difference",
                             choices=["difference", "concatenation", "distance", "hybrid"],
                             help="Embedding combination mode (default: difference)")

    # V3-specific
    v3_group = parser.add_argument_group("V3 Multi-Stream Options")
    v3_group.add_argument("--streams", type=str, default="seeking,driving,profile",
                          help="Active streams (comma-separated, default: seeking,driving,profile)")
    v3_group.add_argument("--n-trajs-per-stream", type=int, default=5,
                          help="Trajectories per stream per branch (default: 5)")
    v3_group.add_argument("--traj-projection-dim", type=int, default=48,
                          help="Per-trajectory projection dimension (default: 48)")
    v3_group.add_argument("--profile-hidden-dims", type=str, default="64,32",
                          help="Profile FCN hidden dims (default: 64,32)")
    v3_group.add_argument("--profile-output-dim", type=int, default=8,
                          help="Profile embedding dimension (default: 8)")
    v3_group.add_argument("--n-profile-features", type=int, default=11,
                          help="Profile feature input dimension (default: 11)")

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100,
                             help="Number of training epochs (default: 100)")
    train_group.add_argument("--batch-size", type=int, default=32,
                             help="Batch size (default: 32)")
    train_group.add_argument("--lr", type=float, default=1e-3,
                             help="Learning rate (default: 0.001)")
    train_group.add_argument("--weight-decay", type=float, default=1e-4,
                             help="Weight decay (default: 0.0001)")
    train_group.add_argument("--early-stopping", type=int, default=10,
                             help="Early stopping patience (default: 10)")
    train_group.add_argument("--scheduler", type=str, default="plateau",
                             choices=["plateau", "cosine", "none"],
                             help="LR scheduler (default: plateau)")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", type=str, default="./checkpoints",
                              help="Output directory for checkpoints (default: ./checkpoints)")
    output_group.add_argument("--experiment-name", type=str,
                              help="Name for this experiment")
    output_group.add_argument("--save-all", action="store_true",
                              help="Save checkpoint at every epoch")

    # Other
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--device", type=str, default="auto",
                             choices=["auto", "cuda", "cpu"],
                             help="Device to use (default: auto)")
    other_group.add_argument("--num-workers", type=int, default=0,
                             help="DataLoader workers (default: 0)")
    other_group.add_argument("--seed", type=int, default=42,
                             help="Random seed (default: 42)")
    other_group.add_argument("--quiet", action="store_true",
                             help="Suppress verbose output")

    return parser.parse_args()


def load_data(args):
    """Load training and validation datasets based on arguments."""
    if args.data_dir:
        datasets = load_dataset_from_directory(args.data_dir)
        train_dataset = datasets['train']
        val_dataset = datasets['val']

    elif args.train and args.val:
        train_dataset = TrajectoryPairDataset(args.train)
        val_dataset = TrajectoryPairDataset(args.val)

    elif args.data:
        full_dataset = TrajectoryPairDataset(args.data)
        train_dataset, val_dataset = create_train_val_split(
            full_dataset,
            val_ratio=args.val_split,
            seed=args.seed
        )

    else:
        raise ValueError("Must provide --data, --data-dir, or both --train and --val")

    return train_dataset, val_dataset


def create_model(args):
    """Create model based on version and arguments."""
    lstm_hidden_dims = tuple(int(x) for x in args.lstm_hidden_dims.split(","))
    classifier_dims = tuple(int(x) for x in args.classifier_dims.split(","))
    bidirectional = not args.no_bidirectional

    if args.model_version == "v3":
        streams = tuple(s.strip() for s in args.streams.split(","))
        profile_hidden_dims = tuple(int(x) for x in args.profile_hidden_dims.split(","))

        model = MultiStreamSiameseDiscriminator(
            lstm_hidden_dims=lstm_hidden_dims,
            dropout=args.dropout,
            bidirectional=bidirectional,
            classifier_hidden_dims=classifier_dims,
            combination_mode=args.combination_mode,
            n_profile_features=args.n_profile_features,
            profile_hidden_dims=profile_hidden_dims,
            profile_output_dim=args.profile_output_dim,
            streams=streams,
            n_trajs_per_stream=args.n_trajs_per_stream,
            traj_projection_dim=args.traj_projection_dim,
        )

    elif args.model_version == "v2":
        model = SiameseLSTMDiscriminatorV2(
            lstm_hidden_dims=lstm_hidden_dims,
            dropout=args.dropout,
            bidirectional=bidirectional,
            classifier_hidden_dims=classifier_dims,
            combination_mode=args.combination_mode,
        )

    else:  # v1
        model = SiameseLSTMDiscriminator(
            lstm_hidden_dims=lstm_hidden_dims,
            dropout=args.dropout,
            bidirectional=bidirectional,
            classifier_hidden_dims=classifier_dims,
        )

    return model


def main():
    args = parse_args()

    # Load data
    if not args.quiet:
        print("Loading data...")
    train_dataset, val_dataset = load_data(args)

    if not args.quiet:
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        if hasattr(train_dataset, 'get_stats'):
            stats = train_dataset.get_stats()
            for k, v in stats.items():
                if k != 'n_samples':
                    print(f"  {k}: {v}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    model = create_model(args)

    if not args.quiet:
        print(f"\nModel architecture:")
        print(f"  Version: {args.model_version}")
        if hasattr(model, 'config'):
            for k, v in model.config.items():
                print(f"  {k}: {v}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")

    # Create training config
    config = TrainingConfig(
        lstm_hidden_dims=tuple(int(x) for x in args.lstm_hidden_dims.split(",")),
        dropout=args.dropout,
        bidirectional=not args.no_bidirectional,
        classifier_hidden_dims=tuple(int(x) for x in args.classifier_dims.split(",")),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        scheduler=args.scheduler,
        checkpoint_dir=args.output,
        save_best_only=not args.save_all,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Build dataset info for logging
    dataset_info = {}
    if args.data_dir:
        dataset_info['data_dir'] = args.data_dir
    if hasattr(train_dataset, 'get_stats'):
        dataset_info['train_stats'] = train_dataset.get_stats()
    if hasattr(val_dataset, 'get_stats'):
        dataset_info['val_stats'] = val_dataset.get_stats()

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=args.experiment_name,
        dataset_info=dataset_info,
    )

    # Train
    history = trainer.train(verbose=not args.quiet)

    if not args.quiet:
        print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
        print("  - best.pt: Best model checkpoint")
        print("  - latest.pt: Most recent checkpoint")
        print("  - config.json: Training configuration")
        print("  - history.json: Training history")


if __name__ == "__main__":
    main()
