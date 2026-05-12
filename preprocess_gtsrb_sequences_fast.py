"""Fast preprocessing: create pseudo-sequence .npz files from `archive/Train` only.

Use this to quickly populate `data/preprocessed_sequences/` so feature precomputation
can be run separately (and optionally on GPU) later.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
from config import DatasetConfig

ARCHIVE_TRAIN_DIR = Path("archive/Train")
OUTPUT_SEQ_DIR = Path("data/preprocessed_sequences")
SPLIT = "train"
SEQ_LEN = DatasetConfig.SEQUENCE_LENGTH
STRIDE = DatasetConfig.FRAME_STRIDE
RESIZE = DatasetConfig.IMAGE_SIZE
FPS = 30.0


def gather_class_dirs(archive_dir: Path):
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive train dir not found: {archive_dir}")
    class_dirs = [p for p in sorted(archive_dir.iterdir()) if p.is_dir()]
    return class_dirs


def make_sequences_from_class(class_dir: Path, class_label: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in ('.ppm', '.png', '.jpg', '.jpeg')])
    num = len(image_files)
    count = 0
    for start in range(0, num - SEQ_LEN + 1, STRIDE or 1):
        end = start + SEQ_LEN
        seq_files = image_files[start:end]
        frames = []
        for f in seq_files:
            img = Image.open(f).convert('RGB')
            if RESIZE:
                img = img.resize(RESIZE)
            frames.append(np.array(img, dtype=np.uint8))
        frames_arr = np.stack(frames)
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
        out_path = out_dir / SPLIT / f"class_{class_label:02d}"
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path / f"{metadata['sequence_id']}.npz", frames=frames_arr, metadata=json.dumps(metadata))
        count += 1
    print(f"Saved {count} sequences for class {class_label} to {out_path}")
    return count


def preprocess_all():
    class_dirs = gather_class_dirs(ARCHIVE_TRAIN_DIR)
    total = 0
    for class_dir in class_dirs:
        try:
            class_label = int(class_dir.name)
        except Exception:
            continue
        out_dir = OUTPUT_SEQ_DIR
        made = make_sequences_from_class(class_dir, class_label, out_dir)
        total += made
    print(f"Total sequences created: {total}")
    return total


if __name__ == '__main__':
    print("Creating pseudo-sequences from archive/Train...")
    preprocess_all()
    print("Done.")
