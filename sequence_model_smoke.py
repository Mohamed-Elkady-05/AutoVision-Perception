"""Reusable smoke-test runner for sequence models on GTSRB synthetic fallback data.

This module keeps the training, evaluation, and visualization path identical for
RNN, GRU, LSTM, and Transformer models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from config import SequentialModelConfig, TrainingConfig
from results_visualization import save_confusion_matrix
from sequence_dataset import SequenceDataset, create_sequence_dataloaders
from unified_trainer import UnifiedTrainer


ModelFactory = Callable[[str], torch.nn.Module]


def run_sequence_model_smoke_test(
    model_name: str,
    model_factory: Callable[[str], torch.nn.Module],
    epochs: int = 50,
    batch_size: int = 16,
    data_dir: str | Path = "./cache/vgg16_sequence_features",
    results_dir: str | Path = "results",
    checkpoints_dir: str | Path = "checkpoints",
) -> dict:
    """Run a full training smoke test for a sequence model.

    The loader falls back to synthetic temporal sequences when no real precomputed
    GTSRB sequence features are present, which lets us validate the training stack.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"{model_name.upper()} TRAINING SMOKE TEST")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Sequence data dir: {data_path.resolve()}")

    metadata_dataset = SequenceDataset(
        features_path=str(data_path),
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

    train_loader, val_loader, test_loader = create_sequence_dataloaders(
        features_dir=str(data_path),
        batch_size=batch_size,
        num_workers=0,
        augment=True,
        seed=42,
        return_metadata=False,
    )

    batch_sequences, batch_labels = next(iter(train_loader))
    print("\n[Batch Check]")
    print(f"  Batch sequences shape: {batch_sequences.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")

    model_config = SequentialModelConfig()
    model = model_factory(device)

    training_config = TrainingConfig()
    training_config.NUM_EPOCHS = epochs
    training_config.BATCH_SIZE = batch_size
    training_config.DEVICE = device
    training_config.EARLY_STOPPING_PATIENCE = max(epochs + 1, 100)
    training_config.LOG_INTERVAL = 10

    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        save_dir=str(checkpoints_dir),
    )

    trainer.train(num_epochs=epochs)
    metrics = trainer.evaluate(test_loader)

    trainer.save_training_curves(str(results_path))
    trainer.save_metrics_json(metrics, str(results_path))

    if hasattr(trainer, "last_eval_targets") and hasattr(trainer, "last_eval_preds"):
        save_confusion_matrix(
            trainer.last_eval_targets,
            trainer.last_eval_preds,
            model_name=model.get_model_name(),
            output_dir=results_path,
            normalize=False,
        )

    print("\n[Final Metrics]")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")

    return metrics
