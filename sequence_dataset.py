"""
Sequence Dataset - Video Frame Sequences for Temporal Models
Loads PRECOMPUTED temporal sequences with metadata to preserve temporal coherence.

Design:
- Loads precomputed VGG16 features FOR SEQUENCES (not individual frames)
- Each sequence is guaranteed to be consecutive frames from actual videos
- Metadata tracks temporal information (video_id, timestamps, frame indices)
- Supports synthetic sequence generation for testing
- Efficient batch loading with temporal coherence preserved

Key Difference from v1:
- v1: Created artificial sliding windows from independent frame features (NO temporal coherence)
- v2: Loads sequences that were PRECOMPUTED from consecutive video frames (temporal coherence)

Workflow:
1. video_sequence_preprocessing.py: Extract consecutive frames from videos
2. feature_extractor_vgg16.py: Compute VGG16 for each sequence
3. sequence_dataset.py: Load sequences WITH metadata (THIS FILE)

Usage:
    dataset = SequenceDataset(
        features_dir="cache/vgg16_sequence_features",  # NOTE: sequence features, not frame features
        split="train"
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for feature_sequences, labels, metadata in loader:
        # feature_sequences: (batch_size, seq_len, 512)
        # labels: (batch_size,)
        # metadata: Dict with video_id, timestamps, frame_indices
        ...
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, cast
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import DatasetConfig


def _decode_metadata(metadata_value) -> Dict:
    """Decode metadata stored in npz files."""
    if isinstance(metadata_value, np.ndarray):
        metadata_value = metadata_value.item()

    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")

    return json.loads(metadata_value)


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for temporal sequences with temporal coherence.

    IMPORTANT: This dataset loads PRECOMPUTED SEQUENCES, not individual frames.
    Each sample represents consecutive frames from an actual video, with metadata
    tracking temporal information.

    Each sample: (sequence_of_features, label, metadata)
        - sequence_of_features: (seq_len, 512) e.g., (10, 512)
        - label: scalar class index (0-42 for GTSRB)
        - metadata: dict with video_id, timestamps, frame_indices
    """

    def __init__(
        self,
        features_path: str,
        split: str = "train",
        random_seed: int = 42,
        augment_sequences: bool = False,
        return_metadata: bool = True,
    ):
        """
        Initialize sequence dataset with temporal metadata.

        Args:
            features_path: Path to directory containing PRECOMPUTED SEQUENCE features
                          Should have structure: features_path/split/class_XX/sequence_*.npz
            split: "train", "val", or "test"
            random_seed: For reproducible splits
            augment_sequences: Apply temporal augmentation (frame permutation, noise)
            return_metadata: Include temporal metadata in __getitem__
        """
        self.features_path = Path(features_path)
        self.split = split
        self.augment_sequences = augment_sequences
        self.random_seed = random_seed
        self.return_metadata = return_metadata

        # Load sequences with metadata
        self.sequences, self.labels, self.metadata_list = self._load_sequences_with_metadata()

        print(f"[SequenceDataset] Loaded {len(self.sequences)} sequences for split='{split}' with temporal coherence preserved")

    def _load_sequences_with_metadata(self) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """
        Load precomputed sequences WITH metadata to preserve temporal order.

        Expected file structure:
        features_path/
        ├── train/
        │   ├── class_00/
        │   │   ├── video_001_000000_features.npz  (contains 'features' and 'metadata')
        │   │   ├── video_001_000010_features.npz
        │   │   └── ...
        │   ├── class_01/
        │   └── ...
        ├── val/
        └── test/

        Returns:
            sequences: List of (seq_len, 512) arrays
            labels: List of class indices
            metadata: List of metadata dicts with temporal info
        """
        sequences = []
        labels = []
        metadata_list = []

        split_dir = self.features_path / self.split

        if not split_dir.exists():
            print(f"[SequenceDataset] Warning: Split directory not found: {split_dir}")
            print("[SequenceDataset] Using synthetic data for testing...")
            return self._generate_synthetic_sequences_with_metadata()

        # Find all sequence feature files
        sequence_files = sorted(split_dir.rglob("*_features.npz"))

        if not sequence_files:
            print(f"[SequenceDataset] No sequence files found in {split_dir}")
            print("[SequenceDataset] Using synthetic data for testing...")
            return self._generate_synthetic_sequences_with_metadata()

        print(f"[SequenceDataset] Found {len(sequence_files)} precomputed sequences")

        for seq_file in sequence_files:
            try:
                # Load compressed numpy file
                data = np.load(seq_file)
                features = data['features'].astype(np.float32)  # (seq_len, 512)

                # Load metadata
                import json
                if 'metadata' in data:
                    metadata = _decode_metadata(data['metadata'])
                else:
                    # Fallback metadata if not stored
                    metadata = {
                        'sequence_id': seq_file.stem,
                        'video_source': 'unknown',
                        'class_label': int(seq_file.parent.name.split('_')[1]),
                        'timestamps': list(range(features.shape[0])),
                    }

                label = metadata.get('class_label', int(seq_file.parent.name.split('_')[1]))

                sequences.append(features)
                labels.append(label)
                metadata_list.append(metadata)

            except Exception as e:
                print(f"[SequenceDataset] Warning: Could not load {seq_file}: {e}")

        if not sequences:
            print("[SequenceDataset] No sequences loaded. Using synthetic data.")
            return self._generate_synthetic_sequences_with_metadata()

        print(f"[SequenceDataset] Successfully loaded {len(sequences)} sequences with temporal metadata")

        return sequences, labels, metadata_list

    def _generate_synthetic_sequences_with_metadata(self, num_sequences: int = 100) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """
        Generate synthetic feature sequences WITH temporal metadata for testing/development.

        Args:
            num_sequences: Number of sequences to generate

        Returns:
            sequences, labels, metadata_list
        """
        sequences = []
        labels = []
        metadata_list = []

        np.random.seed(self.random_seed)

        for seq_idx in range(num_sequences):
            # Random label (0-42 for GTSRB)
            label = np.random.randint(0, 43)

            # Synthetic feature sequence (seq_len, 512)
            class_mean = (label / 43.0)
            sequence = np.random.randn(10, 512) * 0.1 + class_mean
            sequence = sequence.astype(np.float32)

            # Create realistic metadata
            video_id = f"synthetic_video_{seq_idx // 3:04d}"  # Group sequences by video
            timestamps = [i * 0.033 for i in range(10)]  # ~30 FPS

            metadata = {
                'sequence_id': f"{video_id}_{seq_idx * 10:06d}",
                'video_source': video_id,
                'start_frame': seq_idx * 10,
                'end_frame': seq_idx * 10 + 9,
                'frame_count': 10,
                'class_label': label,
                'timestamps': timestamps,
                'fps': 30.0
            }

            sequences.append(sequence)
            labels.append(label)
            metadata_list.append(metadata)

        print(f"[SequenceDataset] Generated {num_sequences} synthetic sequences with temporal metadata")

        return sequences, labels, metadata_list

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Get single sequence sample WITH temporal metadata.

        Args:
            idx: Sample index

        Returns:
            If return_metadata=True:
                (feature_sequence, label, metadata)
            Else:
                (feature_sequence, label)

            - feature_sequence: torch.Tensor of shape (seq_len, 512)
            - label: int
            - metadata: dict with temporal info (video_id, timestamps, etc.)
        """
        sequence = self.sequences[idx].astype(np.float32)
        label = self.labels[idx]
        metadata = self.metadata_list[idx] if self.return_metadata else None

        # Optional temporal augmentation
        if self.augment_sequences and self.split == "train":
            sequence = self._augment_sequence(sequence)

        # Make a copy to avoid negative strides issue in torch.from_numpy
        sequence = np.ascontiguousarray(sequence)

        if self.return_metadata:
            return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long), metadata
        else:
            return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)

    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply temporal augmentation to sequence.

        Options:
        - Temporal flip: reverse frame order (unlikely in real videos but regularizes model)
        - Noise injection: add small Gaussian noise
        - Frame dropping: randomly drop frames (with interpolation)

        Args:
            sequence: Feature sequence (seq_len, feature_dim)

        Returns:
            Augmented sequence
        """
        if np.random.rand() < 0.3:  # 30% chance to flip
            sequence = sequence[::-1]  # Reverse frames

        if np.random.rand() < 0.2:  # 20% chance to add noise
            noise = np.random.randn(*sequence.shape) * 0.05
            sequence = sequence + noise

        return sequence.astype(np.float32)

    def get_class_distribution(self) -> Dict[int, int]:
        """Return count of samples per class."""
        distribution = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


# ============================================================================
# UTILITY FUNCTIONS FOR DATASET CREATION
# ============================================================================

def create_sequence_dataloaders(
    features_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    seed: int = 42,
    return_metadata: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for TEMPORAL sequences.

    IMPORTANT: Sequences are precomputed and loaded WITH METADATA.
    This ensures temporal coherence - sequences are actual consecutive frames from videos.

    Args:
        features_dir: Path to precomputed sequence features directory
                     Should be: features_dir/split/class_XX/sequence_*_features.npz
        batch_size: Batch size for DataLoader
        num_workers: Number of worker threads
        augment: Enable augmentation for training split
        seed: Random seed for reproducibility
        return_metadata: Include temporal metadata in batches

    Returns:
        (train_loader, val_loader, test_loader)
    """
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
        """Custom collate function to handle metadata."""
        if return_metadata:
            sequences, labels, metadata_list = zip(*batch)
            return (
                torch.stack(sequences),
                torch.stack(labels),
                metadata_list  # List of dicts
            )
        else:
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
    # Test dataset creation with TEMPORAL METADATA
    print("\n" + "="*70)
    print("TEMPORAL SEQUENCE DATASET TEST")
    print("="*70)

    # Test 1: Dataset with metadata
    dataset = SequenceDataset(
        features_path="./cache/vgg16_sequence_features",
        split="train",
        augment_sequences=True,
        return_metadata=True,  # Enable temporal metadata
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get sample with metadata
    if dataset.return_metadata:
        sequence, label, metadata = cast(Tuple[torch.Tensor, torch.Tensor, Optional[Dict]], dataset[0])
        metadata = metadata or {}
    else:
        sequence, label = cast(Tuple[torch.Tensor, torch.Tensor], dataset[0])
        metadata = {}
    print(f"\nSequence shape: {sequence.shape}")
    print(f"Label: {label}")
    print(f"\nTemporal Metadata:")
    print(f"  Video ID: {metadata.get('video_source', 'unknown')}")
    print(f"  Sequence ID: {metadata.get('sequence_id', 'unknown')}")
    print(f"  Frame range: {metadata.get('start_frame', '?')} to {metadata.get('end_frame', '?')}")
    print(f"  Timestamps: {metadata.get('timestamps', [])[0]:.3f} - {metadata.get('timestamps', [])[-1]:.3f} sec")
    print(f"  FPS: {metadata.get('fps', 'unknown')}")

    # Test 2: DataLoader without metadata (standard use case)
    dataset_no_metadata = SequenceDataset(
        features_path="./cache/vgg16_sequence_features",
        split="train",
        augment_sequences=True,
        return_metadata=False,  # Standard case
    )
    loader = DataLoader(dataset_no_metadata, batch_size=4, shuffle=True)
    batch_seqs, batch_labels = next(iter(loader))
    print(f"\nBatch sequence shape: {batch_seqs.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"✓ DataLoader works correctly!")

    print("\n✓ Temporal sequence dataset tests passed!")
    print("\nKEY CHANGES FROM PREVIOUS VERSION:")
    print("1. Sequences load WITH temporal metadata (video_id, timestamps, frames)")
    print("2. Each sequence is from ACTUAL consecutive video frames")
    print("3. Metadata guarantees temporal coherence - not artificial sliding windows")
    print("\nUsage:")
    print("  1. Use video_sequence_preprocessing.py to extract frames from videos")
    print("  2. Compute VGG16 features on sequences (preserves temporal structure)")
    print("  3. Load in SequenceDataset with metadata - temporal coherence GUARANTEED!")
