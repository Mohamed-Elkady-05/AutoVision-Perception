"""Train and evaluate the GRU sequence model."""

from __future__ import annotations

import argparse

from src.models.sequence_model_runner import run_model_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GRU sequence model")
    parser.add_argument("--features-dir", default="cache/vgg16_sequence_features")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_model_training(
        "gru",
        features_dir=args.features_dir,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()