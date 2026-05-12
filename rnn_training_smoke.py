"""RNN training smoke test for GTSRB sequence loading.

This script is meant to prove the data pipeline and the RNN model work together.
It intentionally uses the synthetic fallback when no real precomputed sequence data
is available, so the loader path, batch shapes, training loop, and evaluation can
be validated in a small run.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from base_sequential_models import RNNModel
from config import SequentialModelConfig, TrainingConfig
from results_visualization import save_confusion_matrix
from sequence_dataset import SequenceDataset
from unified_trainer import UnifiedTrainer


def build_real_data_loaders(data_dir: Path, batch_size: int, seed: int):
    """Build train/val/test loaders from the real precomputed train cache."""
    base_dataset = SequenceDataset(
        features_path=str(data_dir),
        split="train",
        augment_sequences=False,
        return_metadata=False,
    )

    total_size = len(base_dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_dataset = SequenceDataset(
        features_path=str(data_dir),
        split="train",
        augment_sequences=True,
        return_metadata=False,
    )
    val_dataset = SequenceDataset(
        features_path=str(data_dir),
        split="train",
        augment_sequences=False,
        return_metadata=False,
    )
    test_dataset = SequenceDataset(
        features_path=str(data_dir),
        split="train",
        augment_sequences=False,
        return_metadata=False,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_subset.indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_subset.indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_subset.indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("./cache/vgg16_sequence_features")

    print("=" * 80)
    print("RNN TRAINING SMOKE TEST")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Sequence data dir: {data_dir.resolve()}")

    # First inspect metadata-aware loading so the data contract is visible.
    metadata_dataset = SequenceDataset(
        features_path=str(data_dir),
        split="train",
        augment_sequences=False,
        return_metadata=True,
    )
    sample_sequence, sample_label, sample_metadata = metadata_dataset[0]
    print("\n[Data Loading Check]")
    print(f"  Sequence shape: {sample_sequence.shape}")
    print(f"  Label: {sample_label}")
    print(f"  Video source: {sample_metadata.get('video_source')}")
    print(f"  Frame range: {sample_metadata.get('start_frame')} -> {sample_metadata.get('end_frame')}")

    # Build real-data splits from the precomputed train cache.
    train_loader, val_loader, test_loader = build_real_data_loaders(
        data_dir=data_dir,
        batch_size=16,
        seed=42,
    )

    batch_sequences, batch_labels = next(iter(train_loader))
    print("\n[Batch Check]")
    print(f"  Batch sequences shape: {batch_sequences.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")

    model_config = SequentialModelConfig()
    model = RNNModel(
        input_size=model_config.INPUT_SIZE,
        hidden_size=model_config.HIDDEN_SIZE,
        num_layers=model_config.NUM_LAYERS,
        output_size=model_config.OUTPUT_SIZE,
        dropout=model_config.DROPOUT,
        bidirectional=model_config.BIDIRECTIONAL,
        device=device,
    )

    training_config = TrainingConfig()
    training_config.NUM_EPOCHS = 50
    training_config.BATCH_SIZE = 16
    training_config.DEVICE = device
    training_config.EARLY_STOPPING_PATIENCE = 100
    training_config.LOG_INTERVAL = 10

    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        save_dir="checkpoints",
    )

    trainer.train(num_epochs=50)
    metrics = trainer.evaluate(test_loader)

    trainer.save_training_curves("results")
    trainer.save_metrics_json(metrics, "results")

    if hasattr(trainer, "last_eval_targets") and hasattr(trainer, "last_eval_preds"):
        save_confusion_matrix(
            trainer.last_eval_targets,
            trainer.last_eval_preds,
            model_name=model.get_model_name(),
            output_dir="results",
            normalize=False,
        )

    print("\n[Final Metrics]")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
