"""Real GTSRB dataset utilities for training and sequence construction.

The Kaggle archive includes the actual GTSRB images plus CSV manifests.
This module loads the real data directly from the archive and can also build
related pseudo-sequences from real images when temporal models need them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DatasetConfig


@dataclass
class GTSRBRecord:
    """One image entry from the Kaggle GTSRB archive."""

    image_path: Path
    class_id: int
    width: int
    height: int
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int


def _read_csv_records(csv_path: Path, root_dir: Path) -> List[GTSRBRecord]:
    records: List[GTSRBRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = root_dir / row["Path"]
            records.append(
                GTSRBRecord(
                    image_path=image_path,
                    class_id=int(row["ClassId"]),
                    width=int(row.get("Width", 0) or 0),
                    height=int(row.get("Height", 0) or 0),
                    roi_x1=int(row.get("Roi.X1", 0) or 0),
                    roi_y1=int(row.get("Roi.Y1", 0) or 0),
                    roi_x2=int(row.get("Roi.X2", 0) or 0),
                    roi_y2=int(row.get("Roi.Y2", 0) or 0),
                )
            )
    return records


class GTSRBImageDataset(Dataset):
    """Real GTSRB image dataset backed by the Kaggle archive."""

    def __init__(
        self,
        archive_dir: str | Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        records: Optional[Sequence[GTSRBRecord]] = None,
    ):
        self.archive_dir = Path(archive_dir)
        self.split = split.lower()
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(DatasetConfig.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(DatasetConfig.NORMALIZE_MEAN, DatasetConfig.NORMALIZE_STD),
            ]
        )

        if records is not None:
            self.records = list(records)
        else:
            csv_path = self.archive_dir / ("Train.csv" if self.split == "train" else "Test.csv")
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing manifest: {csv_path}")
            self.records = _read_csv_records(csv_path, self.archive_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        record = self.records[index]
        if not record.image_path.exists():
            raise FileNotFoundError(f"Missing image: {record.image_path}")

        image = Image.open(record.image_path).convert("RGB")
        tensor = self.transform(image)
        metadata = {
            "image_path": str(record.image_path),
            "class_id": record.class_id,
            "width": record.width,
            "height": record.height,
            "roi": [record.roi_x1, record.roi_y1, record.roi_x2, record.roi_y2],
            "source": "gtsrb_archive",
        }
        return tensor, torch.tensor(record.class_id, dtype=torch.long), metadata


def build_related_sequence_groups(
    records: Sequence[GTSRBRecord],
    sequence_length: int = 10,
) -> List[List[GTSRBRecord]]:
    """Build related pseudo-sequences from real GTSRB samples.

    GTSRB is an image dataset, not a video dataset. To make sequence models
    meaningful on real GTSRB data, we group samples by class and then by their
    original manifest order. This keeps the samples related while avoiding
    arbitrary cross-class windows.
    """
    grouped: Dict[int, List[GTSRBRecord]] = {}
    for record in records:
        grouped.setdefault(record.class_id, []).append(record)

    sequences: List[List[GTSRBRecord]] = []
    for class_id in sorted(grouped.keys()):
        class_records = grouped[class_id]
        class_records = sorted(class_records, key=lambda item: item.image_path.name)
        for start_index in range(0, len(class_records) - sequence_length + 1, sequence_length):
            window = class_records[start_index : start_index + sequence_length]
            if len(window) == sequence_length:
                sequences.append(window)
    return sequences


class GTSRBSequenceDataset(Dataset):
    """Pseudo-sequence dataset built from real GTSRB image samples."""

    def __init__(
        self,
        archive_dir: str | Path,
        split: str = "train",
        sequence_length: int = 10,
        transform: Optional[transforms.Compose] = None,
    ):
        self.archive_dir = Path(archive_dir)
        self.split = split.lower()
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(DatasetConfig.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(DatasetConfig.NORMALIZE_MEAN, DatasetConfig.NORMALIZE_STD),
            ]
        )

        csv_path = self.archive_dir / ("Train.csv" if self.split == "train" else "Test.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing manifest: {csv_path}")

        self.records = _read_csv_records(csv_path, self.archive_dir)
        self.sequence_groups = build_related_sequence_groups(self.records, sequence_length=sequence_length)

    def __len__(self) -> int:
        return len(self.sequence_groups)

    def __getitem__(self, index: int):
        group = self.sequence_groups[index]
        frames = []
        metadata = {
            "sequence_id": f"{self.split}_{index:06d}",
            "sequence_length": self.sequence_length,
            "source": "gtsrb_archive",
            "class_id": group[0].class_id,
            "image_paths": [],
        }

        for record in group:
            image = Image.open(record.image_path).convert("RGB")
            frames.append(self.transform(image))
            metadata["image_paths"].append(str(record.image_path))

        sequence_tensor = torch.stack(frames, dim=0)
        label = torch.tensor(group[0].class_id, dtype=torch.long)
        return sequence_tensor, label, metadata


def create_gtsrb_dataloaders(
    archive_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    sequence_length: int = 10,
):
    """Create image and sequence data loaders from the Kaggle archive."""
    archive_path = Path(archive_dir)
    train_image_dataset = GTSRBImageDataset(archive_path, split="train")
    test_image_dataset = GTSRBImageDataset(archive_path, split="test")

    train_sequence_dataset = GTSRBSequenceDataset(archive_path, split="train", sequence_length=sequence_length)

    def collate_with_metadata(batch):
        items, labels, metadata = zip(*batch)
        return torch.stack(items), torch.stack(labels), list(metadata)

    train_image_loader = DataLoader(train_image_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_image_loader = DataLoader(test_image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_sequence_loader = DataLoader(
        train_sequence_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_with_metadata,
    )

    return train_image_loader, test_image_loader, train_sequence_loader
