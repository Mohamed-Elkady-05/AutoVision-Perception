"""Reusable visualization helpers for sequential model experiments.

These functions are intentionally model-agnostic so they can be used with
RNN, GRU, LSTM, Transformer, and future sequence models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create the output directory if needed and return it as a Path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_training_curves(
    history: Mapping[str, Sequence[float]],
    model_name: str,
    output_dir: str | Path,
) -> Path:
    """Save loss/accuracy curves for any sequential model."""
    output_path = ensure_output_dir(output_dir)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.get("train_loss", []), label="Train")
    axes[0].plot(epochs, history.get("val_loss", []), label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.get("train_acc", []), label="Train")
    axes[1].plot(epochs, history.get("val_acc", []), label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_path / f"{model_name}_training_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    model_name: str,
    output_dir: str | Path,
    class_labels: Optional[Sequence[str]] = None,
    normalize: bool = False,
) -> Path:
    """Save a confusion matrix plot for any model."""
    output_path = ensure_output_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if normalize else "d")
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_file = output_path / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


def save_metrics_json(metrics: Mapping[str, object], model_name: str, output_dir: str | Path) -> Path:
    """Save metrics to JSON in a serialization-friendly format."""
    output_path = ensure_output_dir(output_dir)
    serializable = {}
    for key, value in metrics.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value

    output_file = output_path / f"{model_name}_metrics.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    return output_file


def save_model_comparison(
    model_metrics: Mapping[str, Mapping[str, float]],
    output_dir: str | Path,
    metric_name: str = "accuracy",
    title: Optional[str] = None,
) -> Path:
    """Plot a comparison chart across multiple models.

    Parameters
    ----------
    model_metrics:
        Mapping from model name to metric dictionary.
    output_dir:
        Destination directory for the figure.
    metric_name:
        Metric key to compare across models.
    title:
        Optional plot title.
    """
    output_path = ensure_output_dir(output_dir)
    model_names = list(model_metrics.keys())
    values = [float(model_metrics[name].get(metric_name, 0.0)) for name in model_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, values, color=["#2563eb", "#0f766e", "#7c3aed", "#b45309"][: len(model_names)])
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title or f"Model Comparison - {metric_name.title()}")
    ax.set_ylim(0, max(values + [1.0]))
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    output_file = output_path / f"model_comparison_{metric_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file
