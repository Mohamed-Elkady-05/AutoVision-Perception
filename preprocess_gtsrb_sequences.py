"""Preprocess GTSRB archive into pseudo-sequences and precompute VGG16 features.

Workflow:
1. Read images under `archive/Train/<class>/`.
2. Build pseudo-sequences per class of length `DatasetConfig.SEQUENCE_LENGTH` with stride `DatasetConfig.FRAME_STRIDE`.
3. Save sequences (frames + metadata) under `data/preprocessed_sequences/<split>/class_XX/` as .npz files.
4. Run `SequenceFeaturePrecomputer.precompute_sequences()` to compute VGG16 features for each sequence and save to `cache/vgg16_sequence_features/<split>/class_XX/` as *_features.npz

Run with:
conda run -n projects python preprocess_gtsrb_sequences.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from config import DatasetConfig
from video_sequence_preprocessing import SequencePreprocessor, SequenceFeaturePrecomputer

# Parameters
ARCHIVE_TRAIN_DIR = Path("archive/Train")
OUTPUT_SEQ_DIR = Path("data/preprocessed_sequences")
FEATURES_OUTPUT_DIR = Path("cache/vgg16_sequence_features")
SPLIT = "train"
SEQ_LEN = DatasetConfig.SEQUENCE_LENGTH
STRIDE = DatasetConfig.FRAME_STRIDE
RESIZE = DatasetConfig.IMAGE_SIZE
FPS = 30.0


def gather_class_dirs(archive_dir: Path) -> List[Path]:
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive train dir not found: {archive_dir}")
    class_dirs = [p for p in sorted(archive_dir.iterdir()) if p.is_dir()]
    return class_dirs


def make_sequences_from_class(class_dir: Path, class_label: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in ('.ppm', '.png', '.jpg', '.jpeg')])
    images = image_files
    num = len(images)
    count = 0
    for start in range(0, num - SEQ_LEN + 1, STRIDE or 1):
        end = start + SEQ_LEN
        seq_files = images[start:end]
        frames = []
        for f in seq_files:
            img = Image.open(f).convert('RGB')
            if RESIZE:
                img = img.resize(RESIZE)
            frames.append(np.array(img, dtype=np.uint8))
        frames_arr = np.stack(frames)  # (seq_len, H, W, 3)
        metadata = {
            'sequence_id': f"class_{class_label:02d}_{start:06d}",
            'video_source': str(class_dir.name),
            'start_frame': start,
            'end_frame': end - 1,
            'frame_count': SEQ_LEN,
            'class_label': class_label,
            'timestamps': [i / FPS for i in range(start, end)],
            'fps': FPS,
        }
        out_path = out_dir / f"{metadata['sequence_id']}.npz"
        np.savez_compressed(out_path, frames=frames_arr, metadata=json.dumps(metadata))
        count += 1
    print(f"Saved {count} sequences for class {class_label} to {out_dir}")
    return count


def preprocess_all():
    class_dirs = gather_class_dirs(ARCHIVE_TRAIN_DIR)
    total = 0
    for class_dir in class_dirs:
        try:
            class_label = int(class_dir.name)
        except Exception:
            # skip non-numeric dirs
            continue
        out_dir = OUTPUT_SEQ_DIR / SPLIT / f"class_{class_label:02d}"
        made = make_sequences_from_class(class_dir, class_label, out_dir)
        total += made
    print(f"Total sequences created: {total}")
    return total


def compute_features():
    precomputer = SequenceFeaturePrecomputer(sequence_dir=str(OUTPUT_SEQ_DIR), output_dir=str(FEATURES_OUTPUT_DIR))
    precomputer.precompute_sequences(split=SPLIT)


if __name__ == '__main__':
    print("Starting GTSRB preprocessing into pseudo-sequences...")
    sequences_created = preprocess_all()
    if sequences_created == 0:
        print("No sequences created. Check that archive/Train has class folders with images.")
    else:
        print("Precomputing VGG16 features for sequences (this may take time)...")
        compute_features()
        print("Feature precomputation complete.")
