"""Render real pseudo-sequence frames with RNN predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import SequentialModelConfig
from src.models.base_sequential_models import RNNModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SEQUENCE_DIR = PROJECT_ROOT / "data" / "preprocessed_sequences" / "train"
FEATURE_SEQUENCE_DIR = PROJECT_ROOT / "cache" / "vgg16_sequence_features" / "train"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "RNN_best.pt"
OUTPUT_PATH = PROJECT_ROOT / "results" / "RNN_sequence_visualizations.png"


def _decode_metadata(metadata_value) -> Dict:
    if isinstance(metadata_value, np.ndarray):
        metadata_value = metadata_value.item()
    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")
    return json.loads(metadata_value)


def _load_sequence_pair(sequence_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    data = np.load(sequence_file, allow_pickle=False)
    frames = data["frames"]
    metadata = _decode_metadata(data["metadata"])

    feature_file = FEATURE_SEQUENCE_DIR / sequence_file.parent.name / f"{sequence_file.stem}_features.npz"
    feature_data = np.load(feature_file, allow_pickle=False)
    features = feature_data["features"].astype(np.float32)

    return frames, features, metadata


def _load_model(device: str) -> RNNModel:
    model_config = SequentialModelConfig()
    model = RNNModel(
        input_size=model_config.INPUT_SIZE,
        hidden_size=model_config.HIDDEN_SIZE,
        num_layers=model_config.NUM_LAYERS,
        output_size=model_config.OUTPUT_SIZE,
        dropout=model_config.DROPOUT,
        bidirectional=model_config.BIDIRECTIONAL,
        device=device,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict(model: RNNModel, features: np.ndarray, device: str) -> Tuple[int, float]:
    with torch.no_grad():
        batch = torch.from_numpy(features).unsqueeze(0).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
    return int(pred_idx.item()), float(confidence.item())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"RNN checkpoint not found: {CHECKPOINT_PATH}")

    model = _load_model(device)

    sequence_files = sorted(RAW_SEQUENCE_DIR.rglob("*.npz"))
    if not sequence_files:
        raise FileNotFoundError(f"No sequence files found in {RAW_SEQUENCE_DIR}")

    sample_files = sequence_files[:6]
    figure_rows = []

    for sequence_file in sample_files:
        frames, features, metadata = _load_sequence_pair(sequence_file)
        predicted_label, confidence = _predict(model, features, device)
        figure_rows.append((frames, metadata, predicted_label, confidence, sequence_file.stem))

    cols = max(row[0].shape[0] for row in figure_rows)
    fig, axes = plt.subplots(len(figure_rows), cols, figsize=(2.0 * cols, 2.8 * len(figure_rows)))
    if len(figure_rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, (frames, metadata, predicted_label, confidence, sequence_id) in enumerate(figure_rows):
        true_label = metadata.get("class_label", "unknown")
        for col_index in range(cols):
            ax = axes[row_index, col_index]
            ax.axis("off")
            if col_index < frames.shape[0]:
                ax.imshow(frames[col_index].astype(np.uint8))
                ax.set_title(f"F{col_index + 1}", fontsize=9)
        axes[row_index, 0].set_ylabel(
            f"true={true_label}\npred={predicted_label}\nconf={confidence:.3f}",
            rotation=0,
            labelpad=36,
            va="center",
            fontsize=9,
        )
        axes[row_index, 0].set_title(sequence_id, fontsize=10)

    fig.suptitle("Real GTSRB Sequence Frames with RNN Predictions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved sequence visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()