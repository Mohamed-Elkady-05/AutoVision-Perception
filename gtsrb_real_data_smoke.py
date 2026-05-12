"""Real-data smoke test for the Kaggle GTSRB archive.

This script proves the repo can load the attached Kaggle archive, build related
real-image sequences, and convert them into VGG16 feature sequences suitable for
the sequential models.
"""

from __future__ import annotations

from pathlib import Path
from itertools import islice

import numpy as np
import torch

from base_sequential_models import GRUModel
from config import SequentialModelConfig
from feature_extractor_vgg16 import VGG16FeatureExtractor
from gtsrb_dataset import GTSRBImageDataset, GTSRBSequenceDataset, create_gtsrb_dataloaders
from results_visualization import save_confusion_matrix
from unified_trainer import UnifiedTrainer


def main() -> None:
    archive_dir = Path("./archive")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("REAL GTSRB DATA SMOKE TEST")
    print("=" * 80)
    print(f"Archive dir: {archive_dir.resolve()}")
    print(f"Device: {device}")

    image_dataset = GTSRBImageDataset(archive_dir, split="train")
    image_tensor, image_label, image_meta = image_dataset[0]
    print("\n[Real Image Sample]")
    print(f"  Image tensor shape: {image_tensor.shape}")
    print(f"  Label: {image_label.item()}")
    print(f"  Path: {image_meta['image_path']}")

    sequence_dataset = GTSRBSequenceDataset(archive_dir, split="train", sequence_length=10)
    sequence_tensor, sequence_label, sequence_meta = sequence_dataset[0]
    print("\n[Real Sequence Sample]")
    print(f"  Sequence tensor shape: {sequence_tensor.shape}")
    print(f"  Label: {sequence_label.item()}")
    print(f"  Sequence id: {sequence_meta['sequence_id']}")
    print(f"  Related image count: {len(sequence_meta['image_paths'])}")

    image_loader, test_loader, sequence_loader = create_gtsrb_dataloaders(
        archive_dir=archive_dir,
        batch_size=8,
        num_workers=0,
        sequence_length=10,
    )

    batch_images, batch_image_labels = next(iter(image_loader))
    batch_sequences, batch_sequence_labels, batch_sequence_meta = next(iter(sequence_loader))
    print("\n[Loader Check]")
    print(f"  Image batch shape: {batch_images.shape}")
    print(f"  Image labels shape: {batch_image_labels.shape}")
    print(f"  Sequence batch shape: {batch_sequences.shape}")
    print(f"  Sequence labels shape: {batch_sequence_labels.shape}")
    print(f"  Sequence metadata entries: {len(batch_sequence_meta)}")

    # Convert a small batch of real sequences into VGG16 features to verify the
    # real-data pipeline used by the sequence models.
    feature_extractor = VGG16FeatureExtractor(device=device)
    feature_batch = []
    label_batch = []

    for sequence_images, label_tensor, _metadata in islice(sequence_loader, 2):
        for sequence_index in range(sequence_images.size(0)):
            images = []
            for frame_index in range(sequence_images.size(1)):
                frame_tensor = sequence_images[sequence_index, frame_index]
                frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                frame_np = ((frame_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]))
                frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
                images.append(frame_np)
            feature_batch.append(feature_extractor.extract_sequence(images))
            label_batch.append(label_tensor[sequence_index].item())

    if feature_batch:
        features = torch.from_numpy(np.stack(feature_batch[:8])).float()
        labels = torch.tensor(label_batch[:8], dtype=torch.long)
        print("\n[Feature Check]")
        print(f"  Feature batch shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")

        model_config = SequentialModelConfig()
        model = GRUModel(
            input_size=model_config.INPUT_SIZE,
            hidden_size=model_config.HIDDEN_SIZE,
            num_layers=model_config.NUM_LAYERS,
            output_size=model_config.OUTPUT_SIZE,
            dropout=model_config.DROPOUT,
            bidirectional=model_config.BIDIRECTIONAL,
            device=device,
        )
        logits = model(features.to(device))
        print("\n[Forward Pass Check]")
        print(f"  Logits shape: {logits.shape}")

    print("\n✓ Real GTSRB archive loading verified")


if __name__ == "__main__":
    main()
