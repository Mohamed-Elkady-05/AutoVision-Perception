"""Train the RNN on the real precomputed sequence cache for 50 epochs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.config import SequentialModelConfig, TrainingConfig
from src.detection.sequence_dataset import SequenceDataset
from src.models.base_sequential_models import RNNModel
from src.models.unified_trainer import UnifiedTrainer
from src.preprocessing.gtsrb_sequence_preprocessing import (
    SequenceFeaturePrecomputer,
    regenerate_grouped_feature_splits,
)


def _save_confusion_matrix(targets, predictions, output_path: Path) -> None:
    matrix = confusion_matrix(targets, predictions)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("RNN Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_real_data_loaders(data_dir: Path, batch_size: int, seed: int):
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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("./cache/vgg16_sequence_features")
    raw_sequence_dir = Path("./data/preprocessed_sequences")

    print("=" * 80)
    print("RNN TRAINING SMOKE TEST")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Sequence data dir: {data_dir.resolve()}")

    if not data_dir.exists() or not list(data_dir.rglob("*_features.npz")):
        print("[Setup] Building feature cache from ./data/preprocessed_sequences...")
        precomputer = SequenceFeaturePrecomputer(sequence_dir=raw_sequence_dir, output_dir=data_dir, device=device)
        precomputer.precompute_sequences(split="train")

    split_missing = any(not list((data_dir / split).rglob("*_features.npz")) for split in ("val", "test"))
    if split_missing:
        print("[Setup] Regenerating leakage-safe grouped train/val/test feature splits...")
        regenerate_grouped_feature_splits(features_root=data_dir, input_split="train", seed=42)

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

    val_count = len(list((data_dir / "val").rglob("*_features.npz")))
    test_count = len(list((data_dir / "test").rglob("*_features.npz")))
    print("\n[Split Check]")
    print(f"  Train files: {len(metadata_dataset)}")
    print(f"  Val files:   {val_count}")
    print(f"  Test files:  {test_count}")

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

    if trainer.last_eval_targets and trainer.last_eval_preds:
        _save_confusion_matrix(
            trainer.last_eval_targets,
            trainer.last_eval_preds,
            Path("results") / f"{model.get_model_name()}_confusion_matrix.png",
        )

    print("\n[Final Metrics]")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()