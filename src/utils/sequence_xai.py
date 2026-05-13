"""Explainability helpers for sequential sequence-classification models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.base_sequential_models import LSTMModel, TransformerModel


@dataclass
class SequenceExplanation:
    """Container for a model explanation artifact."""

    predicted_class: int
    confidence: float
    frame_importance: np.ndarray
    attention_weights: Optional[np.ndarray] = None


def _to_tensor(sequence: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(sequence, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected a (seq_len, features) or (batch, seq_len, features) array, got {tuple(tensor.shape)}"
        )
    return tensor


def _gradients_to_importance(sequence_grad: torch.Tensor) -> np.ndarray:
    importance = sequence_grad.detach().abs().mean(dim=-1).squeeze(0)
    return importance.cpu().numpy().astype(np.float32)


def explain_sequence_model(
    model: torch.nn.Module,
    sequence: np.ndarray,
    device: str = "cpu",
) -> SequenceExplanation:
    """Return class prediction plus a per-frame importance vector."""
    model = model.to(device)

    # First run a deterministic forward in eval mode to get the predicted class
    with torch.no_grad():
        model.eval()
        eval_inputs = _to_tensor(sequence, device)
        logits_eval = model(eval_inputs)
        probabilities = torch.softmax(logits_eval, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_index = int(prediction.item())
    confidence_value = float(confidence.item())

    # For backward on cuDNN RNNs, the model must be in training mode.
    # Re-run the forward with grad enabled in training mode, then backprop.
    model.train()
    inputs = _to_tensor(sequence, device)
    inputs.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    logits[0, predicted_index].backward()
    frame_importance = _gradients_to_importance(inputs.grad)

    attention_weights: Optional[np.ndarray] = None
    if isinstance(model, LSTMModel):
        with torch.no_grad():
            lstm_out, _ = model.lstm(inputs.detach())
            _, weights = model.attention(lstm_out)
            attention_weights = weights.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32)

    return SequenceExplanation(
        predicted_class=predicted_index,
        confidence=confidence_value,
        frame_importance=frame_importance,
        attention_weights=attention_weights,
    )


def save_sequence_explanation(
    explanation: SequenceExplanation,
    output_path: str | Path,
    title: str = "Sequence explanation",
) -> Path:
    """Save a compact bar plot showing frame importance and optional attention."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = np.arange(len(explanation.frame_importance))
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(frames, explanation.frame_importance, color="#2f6fed", alpha=0.9, label="Gradient saliency")

    if explanation.attention_weights is not None:
        axis.plot(frames, explanation.attention_weights, color="#ff7a59", linewidth=2.0, marker="o", label="Attention")

    axis.set_title(f"{title} | class={explanation.predicted_class} | conf={explanation.confidence:.3f}")
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Importance")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def summarize_sequence_explanation(explanation: SequenceExplanation) -> Dict[str, List[float] | int | float]:
    """Convert an explanation into a JSON-serializable dictionary."""
    summary: Dict[str, List[float] | int | float] = {
        "predicted_class": explanation.predicted_class,
        "confidence": explanation.confidence,
        "frame_importance": explanation.frame_importance.tolist(),
    }
    if explanation.attention_weights is not None:
        summary["attention_weights"] = explanation.attention_weights.tolist()
    return summary