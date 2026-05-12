"""Build pseudo-sequences from the extracted GTSRB archive and precompute VGG16 features.

The archive is treated as an ordered image corpus, not as a video dataset:
- images are read from `archive/Train/<class>/`
- files are sorted within each class folder
- fixed-length windows are sliced with stride 2
- each sequence is stored with metadata as a `.npz`
- VGG16 features are then computed for each saved sequence
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from src.config import DatasetConfig
from src.detection.feature_extractor_vgg16 import VGG16FeatureExtractor


ARCHIVE_TRAIN_DIR = Path("archive/Train")
OUTPUT_SEQ_DIR = Path("data/preprocessed_sequences")
FEATURES_OUTPUT_DIR = Path("cache/vgg16_sequence_features")
SPLIT = "train"
SEQ_LEN = DatasetConfig.SEQUENCE_LENGTH
STRIDE = DatasetConfig.FRAME_STRIDE
RESIZE = DatasetConfig.IMAGE_SIZE
FPS = 30.0


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            train_count = max(train_count - 1, 1)
            val_count = max(val_count - 1, 1)

    return train_count, val_count, test_count


def _load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if RESIZE:
        image = image.resize(RESIZE)
    return np.asarray(image, dtype=np.uint8)


class SequencePreprocessor:
    """Create pseudo-sequences from a class folder of ordered still images."""

    def __init__(self, output_dir: Path = OUTPUT_SEQ_DIR, seq_length: int = SEQ_LEN, stride: int = STRIDE):
        self.output_dir = Path(output_dir)
        self.seq_length = seq_length
        self.stride = stride

    def process_class_folder(self, class_dir: Path, class_label: int, split: str = SPLIT) -> int:
        class_dir = Path(class_dir)
        out_dir = self.output_dir / split / f"class_{class_label:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [path for path in class_dir.iterdir() if path.suffix.lower() in (".ppm", ".png", ".jpg", ".jpeg")]
        )

        count = 0
        for start in range(0, len(image_files) - self.seq_length + 1, self.stride or 1):
            end = start + self.seq_length
            sequence_paths = image_files[start:end]
            frames = [_load_image(path) for path in sequence_paths]
            frames_array = np.stack(frames)

            metadata = {
                "sequence_id": f"class_{class_label:02d}_{start:06d}",
                "video_source": class_dir.name,
                "start_frame": start,
                "end_frame": end - 1,
                "frame_count": self.seq_length,
                "class_label": class_label,
                "timestamps": [index / FPS for index in range(start, end)],
                "fps": FPS,
            }

            out_path = out_dir / f"{metadata['sequence_id']}.npz"
            np.savez_compressed(out_path, frames=frames_array, metadata=json.dumps(metadata))
            count += 1

        print(f"Saved {count} sequences for class {class_label} to {out_dir}")
        return count

    def preprocess_all(self, archive_dir: Path = ARCHIVE_TRAIN_DIR, split: str = SPLIT) -> int:
        archive_dir = Path(archive_dir)
        total = 0

        for class_dir in sorted([path for path in archive_dir.iterdir() if path.is_dir()]):
            try:
                class_label = int(class_dir.name)
            except ValueError:
                continue
            total += self.process_class_folder(class_dir, class_label, split=split)

        print(f"Total sequences created: {total}")
        return total


def _load_metadata_value(metadata_value):
    if isinstance(metadata_value, np.ndarray):
        metadata_value = metadata_value.item()
    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")
    return json.loads(metadata_value)


class SequenceFeaturePrecomputer:
    """Precompute VGG16 features for saved sequence `.npz` files."""

    def __init__(self, sequence_dir: Path = OUTPUT_SEQ_DIR, output_dir: Path = FEATURES_OUTPUT_DIR, device: str = "cuda"):
        self.sequence_dir = Path(sequence_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.extractor = VGG16FeatureExtractor(device=device)

    def precompute_sequences(self, split: str = SPLIT) -> int:
        split_dir = self.sequence_dir / split
        if not split_dir.exists():
            print(f"Sequence directory not found: {split_dir}")
            return 0

        sequence_files = sorted(split_dir.rglob("*.npz"))
        if not sequence_files:
            print(f"No sequence files found in {split_dir}")
            return 0

        saved = 0
        for seq_file in sequence_files:
            data = np.load(seq_file, allow_pickle=False)
            frames = data["frames"]
            metadata = _load_metadata_value(data["metadata"]) if "metadata" in data else {}

            frame_images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
            features = self.extractor.extract_sequence(frame_images).astype(np.float32)

            class_folder = seq_file.parent.name
            output_dir = self.output_dir / split / class_folder
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{seq_file.stem}_features.npz"
            np.savez_compressed(output_file, features=features, metadata=json.dumps(metadata))
            saved += 1

        print(f"Saved {saved} feature files to {self.output_dir / split}")
        return saved

    def precompute_all_splits(self, splits: Iterable[str] = ("train", "val", "test")) -> Dict[str, int]:
        saved_by_split: Dict[str, int] = {}
        for split in splits:
            saved_by_split[split] = self.precompute_sequences(split=split)
        return saved_by_split


def regenerate_grouped_feature_splits(
    features_root: Path = FEATURES_OUTPUT_DIR,
    input_split: str = "train",
    sequence_length: int = SEQ_LEN,
    group_size_sequences: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """Create leakage-safe train/val/test feature splits from a single cached split.

    Steps:
    - keep only non-overlapping windows (`start_frame % sequence_length == 0`)
    - group windows by source segment (video_source + chunk id)
    - split by group so related sequences stay in one split
    """
    features_root = Path(features_root)
    source_dir = features_root / input_split
    if not source_dir.exists():
        raise FileNotFoundError(f"Input split not found: {source_dir}")

    stage_dir: Path | None = None
    read_root = source_dir
    if input_split in {"train", "val", "test"}:
        stage_dir = Path(tempfile.mkdtemp(prefix="safe_split_stage_"))
        for feature_file in sorted(source_dir.rglob("*_features.npz")):
            rel_path = feature_file.relative_to(source_dir)
            target_path = stage_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(feature_file, target_path)
        read_root = stage_dir

    split_dirs = {
        "train": features_root / "train",
        "val": features_root / "val",
        "test": features_root / "test",
    }
    for split_dir in split_dirs.values():
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    entries_by_class: Dict[int, List[Tuple[Path, Dict]]] = {}
    feature_files = sorted(read_root.rglob("*_features.npz"))
    for feature_file in feature_files:
        data = np.load(feature_file, allow_pickle=False)
        metadata = _load_metadata_value(data["metadata"]) if "metadata" in data else {}
        class_label = int(metadata.get("class_label", int(feature_file.parent.name.split("_")[1])))
        start_frame = int(metadata.get("start_frame", 0))

        # Drop overlapping windows to prevent cross-split frame reuse leakage.
        if sequence_length > 0 and (start_frame % sequence_length) != 0:
            continue

        entries_by_class.setdefault(class_label, []).append((feature_file, metadata))

    rng = np.random.default_rng(seed)
    copied_counts = {"train": 0, "val": 0, "test": 0}

    for class_label, entries in sorted(entries_by_class.items()):
        groups: Dict[str, List[Tuple[Path, Dict]]] = {}
        for feature_file, metadata in entries:
            start_frame = int(metadata.get("start_frame", 0))
            source = str(metadata.get("video_source", f"class_{class_label:02d}"))
            segment_id = start_frame // max(sequence_length * group_size_sequences, 1)
            group_key = f"{source}::segment_{segment_id}"
            groups.setdefault(group_key, []).append((feature_file, metadata))

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        train_n, val_n, _ = _split_counts(len(group_keys), train_ratio, val_ratio)

        split_by_group: Dict[str, str] = {}
        for index, group_key in enumerate(group_keys):
            if index < train_n:
                split_by_group[group_key] = "train"
            elif index < train_n + val_n:
                split_by_group[group_key] = "val"
            else:
                split_by_group[group_key] = "test"

        for group_key, group_entries in groups.items():
            split = split_by_group[group_key]
            class_dir = split_dirs[split] / f"class_{class_label:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for feature_file, _metadata in group_entries:
                shutil.copy2(feature_file, class_dir / feature_file.name)
                copied_counts[split] += 1

    print("[Split] Regenerated grouped non-overlapping feature cache:")
    print(f"  train: {copied_counts['train']}")
    print(f"  val:   {copied_counts['val']}")
    print(f"  test:  {copied_counts['test']}")
    if stage_dir and stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    return copied_counts


def main() -> None:
    preprocessor = SequencePreprocessor(output_dir=OUTPUT_SEQ_DIR, seq_length=SEQ_LEN, stride=STRIDE)
    sequence_count = preprocessor.preprocess_all()
    if sequence_count == 0:
        print("No sequences created. Check that archive/Train has class folders with images.")
        return

    print("Precomputing VGG16 features for sequences...")
    precomputer = SequenceFeaturePrecomputer(sequence_dir=OUTPUT_SEQ_DIR, output_dir=FEATURES_OUTPUT_DIR)
    precomputer.precompute_sequences(split=SPLIT)
    print("Feature precomputation complete.")


if __name__ == "__main__":
    main()