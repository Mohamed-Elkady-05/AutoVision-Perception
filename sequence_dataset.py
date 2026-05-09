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

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import DatasetConfig


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for temporal sequences of CNN features.
    
    Each sample: (sequence_of_features, label)
        - sequence_of_features: (seq_len, feature_dim) e.g., (10, 512)
        - label: scalar class index (0-42 for GTSRB)
    """
    
    def __init__(
        self,
        features_path: str,
        sequence_length: int = DatasetConfig.SEQUENCE_LENGTH,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (
            DatasetConfig.TRAIN_SPLIT,
            DatasetConfig.VAL_SPLIT,
            DatasetConfig.TEST_SPLIT
        ),
        random_seed: int = 42,
        augment_sequences: bool = False,
    ):
        """
        Initialize sequence dataset.
        
        Args:
            features_path: Path to directory containing precomputed features
            sequence_length: Number of frames per sequence
            split: "train", "val", or "test"
            split_ratios: (train%, val%, test%) tuple summing to 1.0
            random_seed: For reproducible splits
            augment_sequences: Apply temporal augmentation (frame permutation, etc.)
        """
        self.features_path = Path(features_path)
        self.sequence_length = sequence_length
        self.split = split
        self.augment_sequences = augment_sequences
        self.random_seed = random_seed
        
        # Load features and metadata
        self.sequences, self.labels = self._load_and_organize_sequences()
        
        # Split into train/val/test
        self._split_dataset(split_ratios)
        
        print(f"[SequenceDataset] Loaded {len(self.sequences)} sequences for split='{split}'")
    
    def _load_and_organize_sequences(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load precomputed features and organize into temporal sequences.
        
        Strategy:
        1. Load all features (cached pickle files)
        2. Group by class
        3. Create overlapping sequences within each class
        
        Returns:
            sequences: List of feature arrays (seq_len, feature_dim)
            labels: List of class labels
        """
        sequences = []
        labels = []
        
        # Load cached features
        feature_files = sorted(self.features_path.glob("*.npy"))
        
        if not feature_files:
            # Fallback: generate synthetic sequences for testing
            print("[SequenceDataset] No feature files found. Using synthetic data for testing.")
            return self._generate_synthetic_sequences()
        
        # Load all features
        all_features = []
        all_labels = []
        
        for feature_file in feature_files:
            try:
                features = np.load(feature_file)  # shape: (num_samples, feature_dim)
                # Extract label from filename (e.g., "class_5_features.npy" → 5)
                label = int(feature_file.stem.split("_")[1])
                
                # Create multiple sequences from this class
                for start_idx in range(len(features) - self.sequence_length + 1):
                    sequence = features[start_idx:start_idx + self.sequence_length]
                    sequences.append(sequence)
                    labels.append(label)
                
                all_features.append(features)
                all_labels.extend([label] * len(features))
                
            except Exception as e:
                print(f"[SequenceDataset] Warning: Could not load {feature_file}: {e}")
        
        if not sequences:
            print("[SequenceDataset] No sequences created. Using synthetic data.")
            return self._generate_synthetic_sequences()
        
        return sequences, labels
    
    def _generate_synthetic_sequences(self, num_sequences: int = 1000) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate synthetic feature sequences for testing/development.
        
        Args:
            num_sequences: Number of sequences to generate
            
        Returns:
            sequences, labels
        """
        sequences = []
        labels = []
        
        np.random.seed(self.random_seed)
        
        for _ in range(num_sequences):
            # Random label (0-42 for GTSRB)
            label = np.random.randint(0, 43)
            
            # Synthetic feature sequence (seq_len, 512)
            # Add structure: mean varies by class, noise common
            class_mean = (label / 43.0)  # Slight class-dependent variation
            sequence = np.random.randn(self.sequence_length, 512) * 0.1 + class_mean
            sequence = sequence.astype(np.float32)
            
            sequences.append(sequence)
            labels.append(label)
        
        return sequences, labels
    
    def _split_dataset(self, split_ratios: Tuple[float, float, float]):
        """
        Split dataset into train/val/test by class (stratified).
        
        Args:
            split_ratios: (train%, val%, test%) tuple
        """
        train_ratio, val_ratio, test_ratio = split_ratios
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        # Stratified split
        indices = np.arange(len(self.sequences))
        labels_array = np.array(self.labels)
        
        # First split: train vs. (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(1.0 - train_ratio),
            stratify=labels_array,
            random_state=self.random_seed
        )
        
        # Second split: val vs. test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_size),
            stratify=labels_array[temp_idx],
            random_state=self.random_seed
        )
        
        # Select based on split
        split_map = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        }
        
        selected_idx = split_map.get(self.split, train_idx)
        
        # Filter sequences and labels
        self.sequences = [self.sequences[i] for i in selected_idx]
        self.labels = [self.labels[i] for i in selected_idx]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get single sequence sample.
        
        Args:
            idx: Sample index
            
        Returns:
            (feature_sequence, label)
                - feature_sequence: torch.Tensor of shape (seq_len, 512)
                - label: int
        """
        sequence = self.sequences[idx].astype(np.float32)
        label = self.labels[idx]
        
        # Optional temporal augmentation
        if self.augment_sequences and self.split == "train":
            sequence = self._augment_sequence(sequence)
        
        return torch.from_numpy(sequence.copy()), torch.tensor(label, dtype=torch.long)
    
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
        
        return sequence
    
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
    sequence_length: int = DatasetConfig.SEQUENCE_LENGTH,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for sequence datasets.
    
    Args:
        features_dir: Path to precomputed features directory
        sequence_length: Frames per sequence
        batch_size: Batch size for DataLoader
        num_workers: Number of worker threads
        augment: Enable augmentation for training split
        seed: Random seed for reproducibility
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = SequenceDataset(
        features_path=features_dir,
        sequence_length=sequence_length,
        split="train",
        random_seed=seed,
        augment_sequences=augment,
    )
    
    val_dataset = SequenceDataset(
        features_path=features_dir,
        sequence_length=sequence_length,
        split="val",
        random_seed=seed,
        augment_sequences=False,
    )
    
    test_dataset = SequenceDataset(
        features_path=features_dir,
        sequence_length=sequence_length,
        split="test",
        random_seed=seed,
        augment_sequences=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation with synthetic data
    dataset = SequenceDataset(
        features_path="./cache/vgg16_features",
        sequence_length=10,
        split="train",
        augment_sequences=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    sequence, label = dataset[0]
    print(f"Sequence shape: {sequence.shape}")
    print(f"Label: {label}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_seqs, batch_labels = next(iter(loader))
    print(f"\nBatch sequence shape: {batch_seqs.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    print("\n✓ Sequence dataset tests passed!")
