"""
Sequence Dataset - Video Frame Sequences for Temporal Models
Creates sequences of traffic sign frames from the GTSRB dataset.

Design:
- Loads precomputed VGG16 features (or computes on-the-fly)
- Groups frames into temporal sequences
- Supports synthetic sequence generation for testing
- Efficient batch loading with caching

Usage:
    dataset = SequenceDataset(
        features_dir="cache/vgg16_features",
        sequence_length=10,
        split="train"
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for feature_sequences, labels in loader:
        # feature_sequences: (batch_size, seq_len, 512)
        # labels: (batch_size,)
        ...
"""

"""Sequence dataset utilities for real and synthetic temporal features."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import DatasetConfig


def _decode_metadata(metadata_value) -> Dict:
    if isinstance(metadata_value, np.ndarray):
        metadata_value = metadata_value.item()
    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")
    return json.loads(metadata_value)


class SequenceDataset(Dataset):
    """PyTorch dataset for cached temporal sequence features."""

    def __init__(
        self,
        features_path: str,
        split: str = "train",
        random_seed: int = 42,
        augment_sequences: bool = False,
        return_metadata: bool = True,
    ):
        self.features_path = Path(features_path)
        self.split = split
        self.random_seed = random_seed
        self.augment_sequences = augment_sequences
        self.return_metadata = return_metadata

        self.sequences, self.labels, self.metadata_list = self._load_sequences_with_metadata()
        print(f"[SequenceDataset] Loaded {len(self.sequences)} sequences for split='{split}'")

    def _load_sequences_with_metadata(self) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        metadata_list: List[Dict] = []

        split_dir = self.features_path / self.split
        if not split_dir.exists():
            print(f"[SequenceDataset] Warning: Split directory not found: {split_dir}")
            print("[SequenceDataset] Using synthetic data for testing...")
            return self._generate_synthetic_sequences_with_metadata()

        sequence_files = sorted(split_dir.rglob("*_features.npz"))
        if not sequence_files:
            print(f"[SequenceDataset] No feature files found in {split_dir}")
            print("[SequenceDataset] Using synthetic data for testing...")
            return self._generate_synthetic_sequences_with_metadata()

        for seq_file in sequence_files:
            try:
                data = np.load(seq_file, allow_pickle=False)
                features = data["features"].astype(np.float32)

                if "metadata" in data:
                    metadata = _decode_metadata(data["metadata"])
                else:
                    metadata = {
                        "sequence_id": seq_file.stem,
                        "video_source": "unknown",
                        "class_label": int(seq_file.parent.name.split("_")[1]),
                        "timestamps": list(range(features.shape[0])),
                    }

                label = int(metadata.get("class_label", int(seq_file.parent.name.split("_")[1])))
                sequences.append(features)
                labels.append(label)
                metadata_list.append(metadata)
            except Exception as error:
                print(f"[SequenceDataset] Warning: Could not load {seq_file}: {error}")

        if not sequences:
            print("[SequenceDataset] No sequences loaded. Using synthetic data.")
            return self._generate_synthetic_sequences_with_metadata()

        return sequences, labels, metadata_list

    def _generate_synthetic_sequences_with_metadata(self, num_sequences: int = 100) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        metadata_list: List[Dict] = []

        np.random.seed(self.random_seed)

        for seq_idx in range(num_sequences):
            label = int(np.random.randint(0, DatasetConfig.NUM_CLASSES))
            class_mean = label / float(DatasetConfig.NUM_CLASSES)
            sequence = np.random.randn(DatasetConfig.SEQUENCE_LENGTH, 512) * 0.1 + class_mean
            sequence = sequence.astype(np.float32)

            video_id = f"synthetic_video_{seq_idx // 3:04d}"
            timestamps = [i * 0.033 for i in range(DatasetConfig.SEQUENCE_LENGTH)]

            metadata = {
                "sequence_id": f"{video_id}_{seq_idx * 10:06d}",
                "video_source": video_id,
                "start_frame": seq_idx * 10,
                "end_frame": seq_idx * 10 + DatasetConfig.SEQUENCE_LENGTH - 1,
                "frame_count": DatasetConfig.SEQUENCE_LENGTH,
                "class_label": label,
                "timestamps": timestamps,
                "fps": 30.0,
            }

            sequences.append(sequence)
            labels.append(label)
            metadata_list.append(metadata)

        print(f"[SequenceDataset] Generated {num_sequences} synthetic sequences with temporal metadata")
        return sequences, labels, metadata_list

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx].astype(np.float32)
        label = self.labels[idx]
        metadata = self.metadata_list[idx] if self.return_metadata else None

        if self.augment_sequences and self.split == "train":
            sequence = self._augment_sequence(sequence)

        sequence = np.ascontiguousarray(sequence)

        if self.return_metadata:
            return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long), metadata
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)

    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.3:
            sequence = sequence[::-1]
        if np.random.rand() < 0.2:
            noise = np.random.randn(*sequence.shape) * 0.05
            sequence = sequence + noise
        return sequence.astype(np.float32)

    def get_class_distribution(self) -> Dict[int, int]:
        distribution: Dict[int, int] = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


def create_sequence_dataloaders(
    features_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    seed: int = 42,
    return_metadata: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SequenceDataset(
        features_path=features_dir,
        split="train",
        random_seed=seed,
        augment_sequences=augment,
        return_metadata=return_metadata,
    )
    val_dataset = SequenceDataset(
        features_path=features_dir,
        split="val",
        random_seed=seed,
        augment_sequences=False,
        return_metadata=return_metadata,
    )
    test_dataset = SequenceDataset(
        features_path=features_dir,
        split="test",
        random_seed=seed,
        augment_sequences=False,
        return_metadata=return_metadata,
    )

    def collate_with_metadata(batch):
        if return_metadata:
            sequences, labels, metadata_list = zip(*batch)
            return torch.stack(sequences), torch.stack(labels), metadata_list
        sequences, labels = zip(*batch)
        return torch.stack(sequences), torch.stack(labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_with_metadata if return_metadata else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_metadata if return_metadata else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_metadata if return_metadata else None,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEMPORAL SEQUENCE DATASET TEST")
    print("=" * 70)

    dataset = SequenceDataset(
        features_path="./cache/vgg16_sequence_features",
        split="train",
        augment_sequences=True,
        return_metadata=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    if dataset.return_metadata:
        sequence, label, metadata = cast(Tuple[torch.Tensor, torch.Tensor, Optional[Dict]], dataset[0])
        metadata = metadata or {}
    else:
        sequence, label = cast(Tuple[torch.Tensor, torch.Tensor], dataset[0])
        metadata = {}

    print(f"\nSequence shape: {sequence.shape}")
    print(f"Label: {label}")
    print("\nTemporal Metadata:")
    print(f"  Video ID: {metadata.get('video_source', 'unknown')}")
    print(f"  Sequence ID: {metadata.get('sequence_id', 'unknown')}")
    print(f"  Frame range: {metadata.get('start_frame', '?')} to {metadata.get('end_frame', '?')}")
    timestamps = metadata.get('timestamps', [0.0, 0.0])
    print(f"  Timestamps: {timestamps[0]:.3f} - {timestamps[-1]:.3f} sec")
    print(f"  FPS: {metadata.get('fps', 'unknown')}")

    dataset_no_metadata = SequenceDataset(
        features_path="./cache/vgg16_sequence_features",
        split="train",
        augment_sequences=True,
        return_metadata=False,
    )
    loader = DataLoader(dataset_no_metadata, batch_size=4, shuffle=True)
    batch_seqs, batch_labels = next(iter(loader))
    print(f"\nBatch sequence shape: {batch_seqs.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print("✓ DataLoader works correctly!")

    print("\n✓ Temporal sequence dataset tests passed!")
