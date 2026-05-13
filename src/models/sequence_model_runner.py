"""Shared training and inference utilities for sequence models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import LSTMConfig, SequentialModelConfig, TransformerConfig
from src.detection.sequence_dataset import SequenceDataset
from src.models.base_sequential_models import GRUModel, LSTMModel, RNNModel, TransformerModel
from src.models.unified_trainer import UnifiedTrainer
from src.preprocessing.gtsrb_sequence_preprocessing import regenerate_grouped_feature_splits
from src.utils.sequence_xai import explain_sequence_model, save_sequence_explanation, summarize_sequence_explanation


MODEL_SPECS = {
    "rnn": (RNNModel, SequentialModelConfig),
    "gru": (GRUModel, SequentialModelConfig),
    "lstm": (LSTMModel, LSTMConfig),
    "transformer": (TransformerModel, TransformerConfig),
}


def _get_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(model_name: str, device: str) -> torch.nn.Module:
    model_key = model_name.lower()
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(MODEL_SPECS)}")

    model_class, config_class = MODEL_SPECS[model_key]
    config = config_class()

    if model_key == "transformer":
        model = model_class(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_TRANSFORMER_LAYERS,
            output_size=config.OUTPUT_SIZE,
            dropout=config.DROPOUT,
            attention_heads=config.ATTENTION_HEADS,
            ffn_dim=config.FFN_DIM,
            max_seq_len=config.SEQUENCE_LENGTH,
            device=device,
        )
    else:
        model = model_class(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            output_size=config.OUTPUT_SIZE,
            dropout=config.DROPOUT,
            bidirectional=config.BIDIRECTIONAL,
            device=device,
        )

    return model


def ensure_explicit_splits(features_dir: str | Path, seed: int = 42) -> None:
    """Regenerate train/val/test folders if any explicit split is missing."""
    features_root = Path(features_dir)
    split_dirs = [features_root / split for split in ("train", "val", "test")]
    if all(split_dir.exists() and list(split_dir.rglob("*_features.npz")) for split_dir in split_dirs):
        return

    regenerate_grouped_feature_splits(features_root=features_root, input_split="train", seed=seed)


def _group_key(metadata: Dict, sequence_length: int, group_size_sequences: int) -> str:
    video_source = str(metadata.get("video_source", "unknown"))
    start_frame = int(metadata.get("start_frame", 0))
    segment = start_frame // max(sequence_length * group_size_sequences, 1)
    return f"{video_source}::segment_{segment}"


def validate_split_integrity(
    features_dir: str | Path,
    sequence_length: int = 10,
    group_size_sequences: int = 5,
) -> Dict[str, Dict[str, int]]:
    """Check that split folders are present and that grouped sources do not overlap."""
    features_root = Path(features_dir)
    summary: Dict[str, Dict[str, int]] = {}
    group_to_splits: Dict[str, set[str]] = {}
    seen_sequence_ids: Dict[str, str] = {}

    for split in ("train", "val", "test"):
        split_dir = features_root / split
        split_files = sorted(split_dir.rglob("*_features.npz"))
        if not split_files:
            raise FileNotFoundError(f"Missing or empty split directory: {split_dir}")

        class_counts: Dict[str, int] = {}
        for feature_file in split_files:
            data = np.load(feature_file, allow_pickle=False)
            metadata = json.loads(data["metadata"].item() if isinstance(data["metadata"], np.ndarray) else data["metadata"])
            sequence_id = str(metadata.get("sequence_id", feature_file.stem))
            group_key = _group_key(metadata, sequence_length=sequence_length, group_size_sequences=group_size_sequences)

            previous_split = seen_sequence_ids.get(sequence_id)
            if previous_split is not None and previous_split != split:
                raise ValueError(f"Sequence '{sequence_id}' appears in both '{previous_split}' and '{split}'")
            seen_sequence_ids[sequence_id] = split

            group_to_splits.setdefault(group_key, set()).add(split)

            class_label = str(metadata.get("class_label", feature_file.parent.name))
            class_counts[class_label] = class_counts.get(class_label, 0) + 1

        summary[split] = {
            "files": len(split_files),
            "classes": len(class_counts),
        }

    overlapping_groups = {group: sorted(splits) for group, splits in group_to_splits.items() if len(splits) > 1}
    if overlapping_groups:
        sample_group, splits = next(iter(overlapping_groups.items()))
        raise ValueError(f"Grouped source leakage detected for '{sample_group}' across splits {splits}")

    return summary


def _dataset_label_summary(dataset: SequenceDataset) -> Dict[str, object]:
    labels = np.asarray(dataset.labels, dtype=np.int64)
    if labels.size == 0:
        return {
            "samples": 0,
            "classes": 0,
            "majority_class": -1,
            "majority_baseline_accuracy": 0.0,
            "random_chance_accuracy": 0.0,
            "class_distribution": {},
        }

    class_ids, counts = np.unique(labels, return_counts=True)
    majority_index = int(class_ids[int(np.argmax(counts))])
    majority_baseline = float(np.max(counts) / labels.size)
    random_chance = float(1.0 / max(len(class_ids), 1))
    distribution = {str(int(class_id)): int(count) for class_id, count in zip(class_ids, counts)}

    return {
        "samples": int(labels.size),
        "classes": int(len(class_ids)),
        "majority_class": majority_index,
        "majority_baseline_accuracy": majority_baseline,
        "random_chance_accuracy": random_chance,
        "class_distribution": distribution,
    }


def build_loaders(
    features_dir: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    return_metadata: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ensure_explicit_splits(features_dir, seed=seed)

    train_dataset = SequenceDataset(str(features_dir), split="train", random_seed=seed, augment_sequences=True, return_metadata=return_metadata)
    val_dataset = SequenceDataset(str(features_dir), split="val", random_seed=seed, augment_sequences=False, return_metadata=return_metadata)
    test_dataset = SequenceDataset(str(features_dir), split="test", random_seed=seed, augment_sequences=False, return_metadata=return_metadata)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=len(train_dataset) >= batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def load_trained_model(model_name: str, checkpoint_dir: str | Path, device: str | None = None) -> torch.nn.Module:
    device = _get_device(device)
    model = _build_model(model_name, device=device).to(device)
    checkpoint_path = Path(checkpoint_dir) / f"{model.get_model_name()}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_model_training(
    model_name: str,
    features_dir: str | Path = "cache/vgg16_sequence_features",
    results_dir: str | Path = "results",
    checkpoint_dir: str | Path = "checkpoints",
    num_epochs: int = 50,
    batch_size: int = 16,
    seed: int = 42,
    device: str | None = None,
) -> Dict:
    device = _get_device(device)
    features_dir = Path(features_dir)
    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)

    print(f"[Runner] Model: {model_name}")
    print(f"[Runner] Device: {device}")
    print(f"[Runner] Features: {features_dir.resolve()}")

    ensure_explicit_splits(features_dir, seed=seed)
    integrity_summary = validate_split_integrity(features_dir)
    print(f"[Runner] Split integrity: {integrity_summary}")

    train_dataset = SequenceDataset(str(features_dir), split="train", random_seed=seed, augment_sequences=True, return_metadata=False)
    val_dataset = SequenceDataset(str(features_dir), split="val", random_seed=seed, augment_sequences=False, return_metadata=False)
    test_dataset = SequenceDataset(str(features_dir), split="test", random_seed=seed, augment_sequences=False, return_metadata=False)

    sanity_report = {
        "train": _dataset_label_summary(train_dataset),
        "val": _dataset_label_summary(val_dataset),
        "test": _dataset_label_summary(test_dataset),
    }
    print(f"[Runner] Dataset sanity: {sanity_report}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=len(train_dataset) >= batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = _build_model(model_name, device=device)
    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=str(checkpoint_dir),
    )
    trainer.train(num_epochs=num_epochs)
    metrics = trainer.evaluate(test_loader)

    results_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_training_curves(str(results_dir))
    trainer.save_metrics_json(metrics, str(results_dir))

    sample_dataset = SequenceDataset(str(features_dir), split="test", random_seed=seed, augment_sequences=False, return_metadata=True)
    sample_sequence, _sample_label, sample_metadata = sample_dataset[0]
    explanation = explain_sequence_model(model, sample_sequence.cpu().numpy(), device=device)
    explanation_path = results_dir / f"{model.get_model_name()}_xai.png"
    save_sequence_explanation(explanation, explanation_path, title=f"{model.get_model_name()} explanation")

    explanation_summary = summarize_sequence_explanation(explanation)
    with open(results_dir / f"{model.get_model_name()}_xai.json", "w", encoding="utf-8") as handle:
        json.dump({"metadata": sample_metadata, "sanity_report": sanity_report, **explanation_summary}, handle, indent=2)

    print(f"[Runner] Saved XAI artifact: {explanation_path}")
    return metrics
