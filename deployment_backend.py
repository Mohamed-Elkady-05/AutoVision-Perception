"""
Deployment backend for HuggingFace Spaces.

This module provides a small, backend-first inference layer for GTSRB.
It supports:
- Single-image prediction via the CNN baseline
- Temporal/video prediction via frame sampling + optional sequence model
- Optional loading from local checkpoints or HuggingFace Hub artifacts

The backend is intentionally checkpoint-driven. If no weights are present,
it still launches in demo mode so the Gradio UI can be validated end-to-end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from CNN_model import TrafficSignCNN
from base_sequential_models import create_model
from config import DatasetConfig, SequentialModelConfig
from feature_extractor_vgg16 import VGG16FeatureExtractor

try:
    import cv2
except Exception:  # pragma: no cover - optional at import time for environments without OpenCV
    cv2 = None

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional at import time for environments without huggingface_hub
    hf_hub_download = None


GTSRB_CLASS_NAMES: List[str] = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]


@dataclass
class BackendAssets:
    cnn_checkpoint: Optional[Path]
    sequence_checkpoint: Optional[Path]
    sequence_model_name: str = "gru"
    allow_demo_fallback: bool = True


class GTSRTPredictionBackend:
    """Inference backend for GTSRB image and sequence predictions."""

    def __init__(
        self,
        assets: BackendAssets,
        device: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        self.assets = assets
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_k = top_k
        self.class_names = GTSRB_CLASS_NAMES

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(DatasetConfig.IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

        self.cnn_model = self._load_cnn_model(self.assets.cnn_checkpoint)
        self.sequence_model = self._load_sequence_model(
            self.assets.sequence_checkpoint,
            model_name=self.assets.sequence_model_name,
        )
        self.sequence_feature_extractor = None
        if self.sequence_model is not None:
            self.sequence_feature_extractor = VGG16FeatureExtractor(device=str(self.device))

    @classmethod
    def from_environment(cls) -> "GTSRTPredictionBackend":
        """Create a backend from environment variables used by HuggingFace Spaces."""
        cnn_path = cls._resolve_checkpoint_from_env(
            env_key="GTSRB_CNN_CHECKPOINT",
            hf_repo_env="GTSRB_HF_REPO_ID",
            hf_filename_env="GTSRB_CNN_FILENAME",
            token_env="HF_TOKEN",
        )
        sequence_path = cls._resolve_checkpoint_from_env(
            env_key="GTSRB_SEQUENCE_CHECKPOINT",
            hf_repo_env="GTSRB_HF_REPO_ID",
            hf_filename_env="GTSRB_SEQUENCE_FILENAME",
            token_env="HF_TOKEN",
        )
        sequence_model_name = os.getenv("GTSRB_SEQUENCE_MODEL", "gru")
        return cls(
            assets=BackendAssets(
                cnn_checkpoint=cnn_path,
                sequence_checkpoint=sequence_path,
                sequence_model_name=sequence_model_name,
                allow_demo_fallback=True,
            )
        )

    @staticmethod
    def _resolve_checkpoint_from_env(
        env_key: str,
        hf_repo_env: str,
        hf_filename_env: str,
        token_env: str,
    ) -> Optional[Path]:
        """Resolve a checkpoint path from either local disk or HuggingFace Hub."""
        raw_path = os.getenv(env_key)
        if raw_path:
            checkpoint_path = Path(raw_path)
            if checkpoint_path.exists():
                return checkpoint_path

        repo_id = os.getenv(hf_repo_env)
        filename = os.getenv(hf_filename_env)
        if repo_id and filename and hf_hub_download is not None:
            token = os.getenv(token_env)
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            return Path(downloaded)

        return Path(raw_path) if raw_path else None

    def _load_state_dict(self, model: torch.nn.Module, checkpoint_path: Optional[Path]) -> torch.nn.Module:
        """Load a checkpoint if present, otherwise keep the model in demo mode."""
        model = model.to(self.device)
        model.eval()

        if checkpoint_path is None or not checkpoint_path.exists():
            return model

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "net", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace("module.", "")
            cleaned_state_dict[cleaned_key] = value

        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        return model

    def _load_cnn_model(self, checkpoint_path: Optional[Path]) -> torch.nn.Module:
        """Load the CNN baseline model."""
        model = TrafficSignCNN(num_classes=DatasetConfig.NUM_CLASSES)
        return self._load_state_dict(model, checkpoint_path)

    def _load_sequence_model(
        self,
        checkpoint_path: Optional[Path],
        model_name: str = "gru",
    ) -> Optional[torch.nn.Module]:
        """Load a temporal model when a checkpoint is available."""
        if checkpoint_path is None or not checkpoint_path.exists():
            return None

        model_config = {
            "input_size": SequentialModelConfig.INPUT_SIZE,
            "hidden_size": SequentialModelConfig.HIDDEN_SIZE,
            "num_layers": SequentialModelConfig.NUM_LAYERS,
            "output_size": SequentialModelConfig.OUTPUT_SIZE,
            "dropout": SequentialModelConfig.DROPOUT,
            "bidirectional": SequentialModelConfig.BIDIRECTIONAL,
        }
        model = create_model(model_name, model_config, device=str(self.device))
        return self._load_state_dict(model, checkpoint_path)

    def _to_pil(self, image: Any) -> Image.Image:
        """Convert a Gradio/PIL/numpy image input into a PIL image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def _preprocess_image(self, image: Any) -> torch.Tensor:
        """Convert an input image into a CNN-ready tensor."""
        pil_image = self._to_pil(image)
        tensor = self.image_transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)

    def _topk_from_logits(self, logits: torch.Tensor, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Convert raw logits into a structured prediction payload."""
        top_k = top_k or self.top_k
        probabilities = F.softmax(logits, dim=-1)
        values, indices = torch.topk(probabilities, k=min(top_k, probabilities.shape[-1]), dim=-1)

        predictions = []
        for probability, class_index in zip(values[0].detach().cpu().tolist(), indices[0].detach().cpu().tolist()):
            predictions.append(
                {
                    "class_index": int(class_index),
                    "class_name": self.class_names[int(class_index)],
                    "probability": float(probability),
                }
            )

        best_prediction = predictions[0]
        return {
            "predicted_class_index": best_prediction["class_index"],
            "predicted_class_name": best_prediction["class_name"],
            "confidence": best_prediction["probability"],
            "top_k": predictions,
        }

    @torch.no_grad()
    def predict_image(self, image: Any, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Predict traffic sign class for a single image."""
        tensor = self._preprocess_image(image)
        logits = self.cnn_model(tensor)
        result = self._topk_from_logits(logits, top_k=top_k)
        result["mode"] = "cnn"
        result["input_type"] = "image"
        return result

    def _load_video_frames(self, video_path: str, num_frames: int = 10) -> List[Image.Image]:
        """Sample evenly spaced RGB frames from a video file."""
        if cv2 is None:
            raise RuntimeError("OpenCV is required for video inference but is not installed.")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video contains no frames: {video_path}")

        sample_count = min(num_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        selected_frames = []

        frame_map = {int(index): None for index in frame_indices.tolist()}
        current_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if current_index in frame_map:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected_frames.append(Image.fromarray(rgb_frame))
            current_index += 1

        capture.release()

        if not selected_frames:
            raise RuntimeError("No usable frames were sampled from the input video.")

        return selected_frames

    @torch.no_grad()
    def predict_video(self, video_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Predict a traffic sign class from a video clip."""
        frames = self._load_video_frames(video_path, num_frames=DatasetConfig.SEQUENCE_LENGTH)

        if self.sequence_model is not None and self.sequence_feature_extractor is not None:
            features = self.sequence_feature_extractor.extract_sequence(frames)
            tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            logits = self.sequence_model(tensor)
            result = self._topk_from_logits(logits, top_k=top_k)
            result["mode"] = self.sequence_model.get_model_name()
            result["input_type"] = "video"
            result["frame_count"] = len(frames)
            return result

        # Fallback: use CNN on each sampled frame and average the logits.
        logits_list = []
        for frame in frames:
            tensor = self._preprocess_image(frame)
            logits_list.append(self.cnn_model(tensor))
        logits = torch.stack(logits_list, dim=0).mean(dim=0, keepdim=True)
        result = self._topk_from_logits(logits, top_k=top_k)
        result["mode"] = "cnn_frame_average"
        result["input_type"] = "video"
        result["frame_count"] = len(frames)
        result["note"] = "Sequence checkpoint not configured; averaging frame-level CNN predictions."
        return result

    def model_status(self) -> Dict[str, Any]:
        """Summarize which assets were loaded."""
        return {
            "device": str(self.device),
            "cnn_checkpoint": str(self.assets.cnn_checkpoint) if self.assets.cnn_checkpoint else None,
            "sequence_checkpoint": str(self.assets.sequence_checkpoint) if self.assets.sequence_checkpoint else None,
            "sequence_model_name": self.assets.sequence_model_name,
            "cnn_ready": True,
            "sequence_ready": self.sequence_model is not None,
            "demo_mode": self.assets.allow_demo_fallback and self.assets.cnn_checkpoint is None,
        }


def format_prediction_markdown(result: Dict[str, Any]) -> str:
    """Render a compact Markdown summary for the Gradio UI."""
    lines = [
        f"**Predicted class:** {result['predicted_class_name']} ({result['predicted_class_index']})",
        f"**Confidence:** {result['confidence']:.2%}",
        f"**Mode:** {result.get('mode', 'unknown')}",
    ]
    if result.get("note"):
        lines.append(f"**Note:** {result['note']}")
    return "\n\n".join(lines)
