"""Gradio app for HuggingFace Spaces deployment."""

from __future__ import annotations

import json
from typing import Any, Dict

import gradio as gr

from deployment_backend import GTSRTPredictionBackend, format_prediction_markdown


backend = GTSRTPredictionBackend.from_environment()


def predict_image(image: Any, top_k: int) -> tuple[str, Dict[str, Any]]:
    """Run a single-image prediction and return a markdown summary plus details."""
    result = backend.predict_image(image, top_k=top_k)
    return format_prediction_markdown(result), result


def predict_video(video_path: str, top_k: int) -> tuple[str, Dict[str, Any]]:
    """Run a video prediction and return a markdown summary plus details."""
    if not video_path:
        raise gr.Error("Please upload a video file first.")
    result = backend.predict_video(video_path, top_k=top_k)
    return format_prediction_markdown(result), result


def build_app() -> gr.Blocks:
    """Create the Gradio UI."""
    status = backend.model_status()
    status_json = json.dumps(status, indent=2)

    with gr.Blocks(title="AutoVision Perception - GTSRB Demo") as demo:
        gr.Markdown(
            """
            # AutoVision Perception
            GTSRB traffic sign inference with a Gradio front end and a checkpoint-driven backend.

            The app supports single-image inference immediately and can also process short videos.
            If temporal weights are supplied later, the video path will automatically use the sequence model.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Runtime Status")
                status_box = gr.Code(value=status_json, language="json", label="Loaded assets")
                gr.Markdown(
                    """
                    **Input contract**
                    - Single-image mode expects one traffic sign image.
                    - Video mode expects a short clip of related frames.
                    - The backend keeps prediction logic separate from the UI so HuggingFace weights can be swapped later.
                    """
                )
            with gr.Column(scale=1):
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K predictions")

        with gr.Tab("Single Image"):
            gr.Markdown("Upload one traffic sign image for CNN inference.")
            image_input = gr.Image(type="pil", label="Traffic sign image")
            image_button = gr.Button("Predict image", variant="primary")
            image_summary = gr.Markdown()
            image_details = gr.JSON(label="Prediction details")

            image_button.click(
                fn=predict_image,
                inputs=[image_input, top_k],
                outputs=[image_summary, image_details],
            )

        with gr.Tab("Video / Sequence"):
            gr.Markdown(
                "Upload a short clip of related frames. If a sequence checkpoint is available, the backend will use it; otherwise it averages frame-level CNN predictions."
            )
            video_input = gr.Video(label="Video clip", format="mp4", sources=["upload"])
            video_button = gr.Button("Predict video", variant="primary")
            video_summary = gr.Markdown()
            video_details = gr.JSON(label="Prediction details")

            video_button.click(
                fn=predict_video,
                inputs=[video_input, top_k],
                outputs=[video_summary, video_details],
            )

        gr.Markdown(
            """
            ## Deployment Notes
            Configure the following environment variables in HuggingFace Spaces when you add weights:
            - `GTSRB_CNN_CHECKPOINT`
            - `GTSRB_SEQUENCE_CHECKPOINT`
            - `GTSRB_SEQUENCE_MODEL`
            - `GTSRB_HF_REPO_ID` / `GTSRB_CNN_FILENAME` / `GTSRB_SEQUENCE_FILENAME`
            """
        )

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch()
