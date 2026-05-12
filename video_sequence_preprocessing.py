"""
Video Sequence Preprocessing
Extracts consecutive frames from videos and organizes them for temporal analysis.
Creates metadata to track temporal coherence in sequences.

This ensures that sequences are ACTUALLY consecutive frames from videos,
not artificial sliding windows from independent images.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass, asdict


def _load_metadata_value(metadata_value) -> Dict:
    """Decode metadata stored by np.savez_compressed."""
    if isinstance(metadata_value, np.ndarray):
        metadata_value = metadata_value.item()

    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")

    return json.loads(metadata_value)


@dataclass
class SequenceMetadata:
    """Metadata for a temporal sequence."""
    sequence_id: str              # Unique identifier
    video_source: str             # Video file or source
    start_frame: int              # First frame index in video
    end_frame: int                # Last frame index in video
    frame_count: int              # Number of frames
    class_label: int              # Traffic sign class
    timestamps: List[float]       # Frame timestamps (seconds)
    fps: float                    # Frames per second


class VideoSequenceExtractor:
    """
    Extracts temporal sequences from video files while preserving frame order.

    Usage:
        extractor = VideoSequenceExtractor(seq_length=10, stride=2)
        sequences = extractor.extract_from_video("dashcam.mp4", class_label=5)
        # Returns: List[Tuple[frames, metadata]]
    """

    def __init__(self, seq_length: int = 10, stride: int = 1):
        """
        Initialize extractor.

        Args:
            seq_length: Number of frames per sequence
            stride: Step size between sequences (1=overlapping, seq_length=non-overlapping)
        """
        self.seq_length = seq_length
        self.stride = stride

    def extract_from_video(self, video_path: str, class_label: int,
                          resize: Tuple[int, int] = (32, 32)) -> List[Tuple[np.ndarray, SequenceMetadata]]:
        """
        Extract frame sequences from a video file.

        Args:
            video_path: Path to video file
            class_label: Traffic sign class for this video
            resize: Target frame size (height, width)

        Returns:
            List of (frames_array, metadata) tuples
                - frames_array: (seq_length, height, width, 3)
                - metadata: SequenceMetadata with temporal info
        """
        sequences = []
        video_path = Path(video_path)

        if not video_path.exists():
            print(f"[Extractor] Warning: Video not found: {video_path}")
            return sequences

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Extractor] Processing {video_path.name}: {total_frames} frames @ {fps} FPS")

        # Read all frames
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize
            if resize:
                frame = cv2.resize(frame, (resize[1], resize[0]))

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_idx += 1

        cap.release()

        print(f"[Extractor] Loaded {len(frames)} frames")

        # Extract sequences
        for start_idx in range(0, len(frames) - self.seq_length + 1, self.stride):
            end_idx = start_idx + self.seq_length
            sequence_frames = np.array(frames[start_idx:end_idx], dtype=np.uint8)

            # Create metadata
            timestamps = [start_idx / fps, end_idx / fps]
            frame_timestamps = [i / fps for i in range(start_idx, end_idx)]

            metadata = SequenceMetadata(
                sequence_id=f"{video_path.stem}_{start_idx:06d}",
                video_source=str(video_path),
                start_frame=start_idx,
                end_frame=end_idx - 1,
                frame_count=self.seq_length,
                class_label=class_label,
                timestamps=frame_timestamps,
                fps=fps
            )

            sequences.append((sequence_frames, metadata))

        print(f"[Extractor] Extracted {len(sequences)} sequences (seq_len={self.seq_length}, stride={self.stride})")

        return sequences


class SequencePreprocessor:
    """
    Preprocesses video sequences and prepares them for VGG16 feature extraction.
    Saves organized sequences with metadata.
    """

    def __init__(self, output_dir: str, seq_length: int = 10, stride: int = 1):
        """
        Initialize preprocessor.

        Args:
            output_dir: Where to save preprocessed sequences
            seq_length: Frames per sequence
            stride: Stride between sequences
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seq_length = seq_length
        self.stride = stride
        self.extractor = VideoSequenceExtractor(seq_length, stride)

        self.metadata_list = []

    def process_video(self, video_path: str, class_label: int, split: str = "train") -> int:
        """
        Process a single video and save sequences.

        Args:
            video_path: Path to video file
            class_label: Traffic sign class
            split: Dataset split (train/val/test)

        Returns:
            Number of sequences extracted
        """
        # Extract sequences
        sequences = self.extractor.extract_from_video(video_path, class_label)

        if not sequences:
            return 0

        # Save sequences
        split_dir = self.output_dir / split / f"class_{class_label:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for seq_frames, metadata in sequences:
            # Save frames array
            filename = f"{metadata.sequence_id}.npz"
            filepath = split_dir / filename

            np.savez_compressed(
                filepath,
                frames=seq_frames,
                metadata=json.dumps(asdict(metadata))
            )

            self.metadata_list.append(asdict(metadata))
            count += 1

        print(f"[Preprocessor] Saved {count} sequences to {split_dir}")

        return count

    def save_metadata_index(self):
        """Save global metadata index for quick lookup."""
        index_path = self.output_dir / "sequence_index.json"

        with open(index_path, 'w') as f:
            json.dump(self.metadata_list, f, indent=2)

        print(f"[Preprocessor] Saved metadata index: {index_path}")
        print(f"[Preprocessor] Total sequences: {len(self.metadata_list)}")


class SequenceFeaturePrecomputer:
    """
    Precomputes VGG16 features for video sequences while preserving temporal order.

    This ensures features are computed on actual consecutive frames from videos.
    """

    def __init__(self, sequence_dir: str, output_dir: str):
        """
        Initialize feature precomputer.

        Args:
            sequence_dir: Directory with preprocessed video sequences
            output_dir: Where to save precomputed features
        """
        self.sequence_dir = Path(sequence_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def precompute_sequences(self, split: str = "train"):
        """
        Precompute VGG16 features for all sequences in a split.

        Features will be (seq_len, 512) per sequence, with metadata preserved.

        Args:
            split: Dataset split (train/val/test)
        """
        from feature_extractor_vgg16 import VGG16FeatureExtractor

        extractor = VGG16FeatureExtractor()

        split_dir = self.sequence_dir / split
        output_split_dir = self.output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        # Find all sequence files
        sequence_files = list(split_dir.rglob("*.npz"))
        print(f"[FeaturePrecomputer] Found {len(sequence_files)} sequences in {split}")

        for seq_file in sequence_files:
            # Load sequence
            data = np.load(seq_file)
            frames = data['frames']  # (seq_len, height, width, 3)
            metadata_dict = _load_metadata_value(data['metadata'])

            # Compute VGG16 features
            print(f"[FeaturePrecomputer] Processing {seq_file.name}...")
            features_list = []

            for frame in frames:
                # Frame to PIL Image format
                from PIL import Image
                pil_frame = Image.fromarray(frame.astype(np.uint8))
                feature = extractor.extract_single(pil_frame)  # (512,)
                features_list.append(feature)

            features = np.array(features_list)  # (seq_len, 512)

            # Save features with metadata
            class_label = metadata_dict['class_label']
            class_dir = output_split_dir / f"class_{class_label:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            feature_file = class_dir / f"{metadata_dict['sequence_id']}_features.npz"
            np.savez_compressed(
                feature_file,
                features=features,
                metadata=json.dumps(metadata_dict)
            )

            print(f"  ✓ Saved: {feature_file}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Process video files into temporal sequences.

    Workflow:
    1. Extract frames from videos → save as sequences with metadata
    2. Compute VGG16 features on consecutive frames
    3. Use in SequenceDataset with temporal coherence preserved
    """

    print("\n" + "="*70)
    print("VIDEO SEQUENCE PREPROCESSING EXAMPLE")
    print("="*70)

    # Step 1: Extract sequences from videos
    preprocessor = SequencePreprocessor(
        output_dir="./data/preprocessed_sequences",
        seq_length=10,
        stride=5  # Non-overlapping sequences
    )

    # Example: Process a video file (you would replace with actual video paths)
    # video_path = "dashcam_001.mp4"
    # class_label = 5  # Speed limit 50
    # preprocessor.process_video(video_path, class_label, split="train")
    # preprocessor.save_metadata_index()

    print("\nTo use with real videos:")
    print("1. preprocessor.process_video('video.mp4', class_label=5)")
    print("2. Compute features: feature_precomputer.precompute_sequences()")
    print("3. Load in SequenceDataset with temporal coherence preserved")

    print("\n✓ Preprocessing module ready for production use!")
