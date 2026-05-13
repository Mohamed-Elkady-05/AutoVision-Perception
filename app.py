"""
AutoVision-Perception: GTSRB Traffic Sign Classification
Gradio app for Hugging Face Spaces deployment
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
import json

from src.models.base_sequential_models import RNNModel, GRUModel, LSTMModel, TransformerModel

# Global state
device = "cuda" if torch.cuda.is_available() else "cpu"
models = {}
model_metadata = {}

def load_models():
    """Load all trained models."""
    global models, model_metadata
    
    model_configs = {
        "RNN": RNNModel,
        "GRU": GRUModel,
        "LSTM": LSTMModel,
        "Transformer": TransformerModel,
    }
    
    for model_name, model_class in model_configs.items():
        checkpoint_path = f"checkpoints/{model_name.lower()}/{model_name}_best.pt"
        
        if model_name == "Transformer":
            model = TransformerModel(input_size=512, output_size=43, device=device)
        else:
            model = model_class(
                input_size=512,
                hidden_size=256,
                num_layers=2,
                output_size=43,
                dropout=0.3,
                bidirectional=True,
                device=device,
            )
        
        if Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()
            models[model_name] = model
            
            # Load metrics
            metrics_path = f"results/{model_name}_metrics.json"
            if Path(metrics_path).exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    model_metadata[model_name] = {
                        "accuracy": metrics.get("accuracy", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "f1": metrics.get("f1", 0),
                    }
    
    print(f"✓ Loaded {len(models)} models: {list(models.keys())}")


def predict(model_name: str) -> dict:
    """Get model performance and sample prediction info."""
    if model_name not in models:
        return {"error": f"Model {model_name} not loaded"}
    
    metadata = model_metadata.get(model_name, {})
    
    return {
        "Model": model_name,
        "Test Accuracy": f"{metadata.get('accuracy', 0):.2%}",
        "Precision": f"{metadata.get('precision', 0):.2%}",
        "Recall": f"{metadata.get('recall', 0):.2%}",
        "F1 Score": f"{metadata.get('f1', 0):.2%}",
        "Status": "✓ Ready for inference",
    }


def get_model_info(model_name: str) -> str:
    """Return detailed model information."""
    if model_name not in models:
        return f"Model {model_name} not available"
    
    model = models[model_name]
    params = model.get_num_parameters()
    
    info = f"""
## {model_name} Model

**Architecture**: {model.get_model_name()}  
**Total Parameters**: {params:,}  
**Input Size**: 512 (VGG16 features)  
**Output Classes**: 43 (GTSRB traffic signs)  
**Device**: {device.upper()}

**Test Set Performance**:
"""
    
    metadata = model_metadata.get(model_name, {})
    info += f"""
- **Accuracy**: {metadata.get('accuracy', 0):.2%}
- **Precision**: {metadata.get('precision', 0):.2%}
- **Recall**: {metadata.get('recall', 0):.2%}
- **F1 Score**: {metadata.get('f1', 0):.2%}

**Model Type**:
"""
    
    if model_name == "RNN":
        info += "Vanilla Recurrent Neural Network with bidirectional processing"
    elif model_name == "GRU":
        info += "Gated Recurrent Unit with gating mechanisms for better sequence modeling"
    elif model_name == "LSTM":
        info += "Long Short-Term Memory with attention mechanism for traffic sign sequences"
    elif model_name == "Transformer":
        info += "Transformer encoder with self-attention for parallel sequence processing"
    
    return info


# Load models on startup
print("[AutoVision] Loading models...")
load_models()

# Gradio interface
with gr.Blocks(title="AutoVision-Perception", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚦 AutoVision-Perception
    ### GTSRB Traffic Sign Classification using Sequence Models
    
    This application showcases **4 deep learning models** trained on traffic sign sequences:
    - **RNN** (Vanilla Recurrent Neural Network)
    - **GRU** (Gated Recurrent Unit)
    - **LSTM** (Long Short-Term Memory with Attention)
    - **Transformer** (Self-Attention Architecture)
    
    All models are trained on sequences of 10 VGG16-extracted feature frames to classify traffic signs into 43 categories.
    """)
    
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=list(models.keys()),
            value=list(models.keys())[0] if models else "RNN",
            label="Select Model",
            interactive=True,
        )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Performance")
            performance_output = gr.JSON(label="Metrics", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Model Details")
            info_output = gr.Markdown(label="Information")
    
    gr.Markdown("""
    ---
    ### Dataset & Training
    
    **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)  
    **Sequences**: 10-frame pseudo-sequences extracted from traffic sign videos  
    **Features**: 512-dimensional VGG16 features (precomputed)  
    **Splits**: 2,654 train | 529 validation | 737 test sequences  
    **Classes**: 43 traffic sign categories  
    
    **Key Insight**: The test set differs from validation curves because:
    - Validation set is used during training (early stopping, checkpoint selection)
    - Test set is held out completely for unbiased final evaluation
    - Both sets have different class distributions and data characteristics
    """)
    
    gr.Markdown("""
    ---
    ### Results Summary
    
    | Model | Accuracy | Precision | Recall | F1 Score |
    |-------|----------|-----------|--------|----------|
    | RNN | 84.53% | 79.59% | 80.58% | 78.64% |
    | GRU | 83.18% | 78.89% | 78.30% | 77.01% |
    | LSTM | 83.18% | 77.77% | 79.54% | 77.14% |
    | **Transformer** | **85.75%** | **81.12%** | **81.18%** | **79.49%** |
    
    🏆 **Best Model**: Transformer with 85.75% test accuracy
    """)
    
    gr.Markdown("""
    ---
    ### Usage
    
    1. Select a model from the dropdown above
    2. View its performance metrics and architecture details
    3. All models are ready for inference on traffic sign sequences
    
    **Training Configuration**:
    - Optimizer: Adam (lr=1e-3)
    - Batch Size: 32
    - Max Epochs: 30
    - Early Stopping: 10 epochs patience
    - Scheduler: Cosine annealing
    """)
    
    # Event handler
    model_selector.change(
        fn=predict,
        inputs=model_selector,
        outputs=performance_output,
    )
    
    model_selector.change(
        fn=get_model_info,
        inputs=model_selector,
        outputs=info_output,
    )
    
    # Initial load
    demo.load(
        fn=predict,
        inputs=model_selector,
        outputs=performance_output,
    )
    
    demo.load(
        fn=get_model_info,
        inputs=model_selector,
        outputs=info_output,
    )


if __name__ == "__main__":
    demo.launch(share=False)