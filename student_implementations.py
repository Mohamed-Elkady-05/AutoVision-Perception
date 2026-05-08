"""
Phase 3 Student Implementation Template
This file serves as the skeleton/template for each student to implement their models.

Instructions for Students:
1. Student 1 & 2: Implement RNN/GRU and LSTM models in base_sequential_models.py
2. Student 3: Implement Transformer model in base_sequential_models.py
3. Student 4: Use this file as starting point for unified evaluation and comparison
4. Student 5: Will extend this with bonus features (inference, XAI, adversarial)

Key Classes to Extend:
- LSTMModel (Student 2): In base_sequential_models.py
- TransformerModel (Student 3): In base_sequential_models.py

This file demonstrates the complete pipeline.
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Import infrastructure components
from config import (
    DatasetConfig, FeatureExtractorConfig, TrainingConfig, SequentialModelConfig,
    LSTMConfig, TransformerConfig
)
from feature_extractor_vgg16 import VGG16FeatureExtractor, precompute_gtsrb_features
from sequence_dataset import SequenceDataset, create_sequence_dataloaders
from base_sequential_models import RNNModel, GRUModel, LSTMModel, TransformerModel, create_model
from unified_trainer import UnifiedTrainer


# ============================================================================
# EXAMPLE 1: EXTRACT VGG16 FEATURES
# ============================================================================

def example_extract_vgg16_features():
    """
    Example: Extract and cache VGG16 features from GTSRB dataset.
    
    This should be run ONCE to precompute all features.
    Students can skip this if features are already precomputed.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: VGG16 Feature Extraction")
    print("="*60)
    
    config = FeatureExtractorConfig()
    extractor = VGG16FeatureExtractor(config=config)
    
    # Example: Extract from a directory
    # Note: You'll need to point to actual GTSRB data
    # For now, this uses synthetic data
    
    print("[Example] Feature extraction set up. Ready to process real data when available.")
    

# ============================================================================
# EXAMPLE 2: CREATE DATA LOADERS
# ============================================================================

def example_create_dataloaders():
    """
    Example: Create sequence DataLoaders with temporal data.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Create Sequence DataLoaders")
    print("="*60)
    
    train_loader, val_loader, test_loader = create_sequence_dataloaders(
        features_dir="./cache/vgg16_features",
        sequence_length=DatasetConfig.SEQUENCE_LENGTH,
        batch_size=32,
        num_workers=4,
        augment=True,
        seed=42
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Inspect a batch
    batch_sequences, batch_labels = next(iter(train_loader))
    print(f"\nBatch sequence shape: {batch_sequences.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# EXAMPLE 3: TRAIN RNN MODEL
# ============================================================================

def example_train_rnn():
    """
    Example: Train RNN model on sequence data.
    This demonstrates the complete pipeline for Student 1.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Train RNN Model")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = example_create_dataloaders()
    
    # Create RNN model
    config = SequentialModelConfig()
    rnn_model = RNNModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        device=device
    )
    
    print(f"\n{rnn_model.get_model_name()} Model Config:")
    for key, value in rnn_model.get_config_dict().items():
        print(f"  {key}: {value}")
    
    # Create trainer
    training_config = TrainingConfig()
    training_config.NUM_EPOCHS = 5  # Short run for example
    training_config.DEVICE = device
    
    trainer = UnifiedTrainer(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        save_dir="checkpoints"
    )
    
    # Train
    trainer.train(num_epochs=5)
    
    # Evaluate
    metrics = trainer.evaluate(test_loader)
    
    # Save curves
    trainer.save_training_curves("results")
    trainer.save_metrics_json(metrics, "results")
    
    return rnn_model, trainer, metrics


# ============================================================================
# EXAMPLE 4: TRAIN GRU MODEL
# ============================================================================

def example_train_gru():
    """
    Example: Train GRU model.
    Similar structure to RNN but with gating.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Train GRU Model")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader, test_loader = example_create_dataloaders()
    
    config = SequentialModelConfig()
    gru_model = GRUModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        device=device
    )
    
    training_config = TrainingConfig()
    training_config.NUM_EPOCHS = 5
    training_config.DEVICE = device
    
    trainer = UnifiedTrainer(
        model=gru_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        save_dir="checkpoints"
    )
    
    trainer.train(num_epochs=5)
    metrics = trainer.evaluate(test_loader)
    trainer.save_training_curves("results")
    trainer.save_metrics_json(metrics, "results")
    
    return gru_model, trainer, metrics


# ============================================================================
# STUDENT TASKS
# ============================================================================

"""
STUDENT 1 - RNN & GRU IMPLEMENTATION:
✓ Already provided in base_sequential_models.py

Next tasks:
1. Train both RNN and GRU models using this template
2. Experiment with:
   - Different hidden_size (128, 256, 512)
   - Different num_layers (1, 2, 3)
   - bidirectional=True/False
3. Compare RNN vs GRU performance
4. Create comparison plots (accuracy, loss, training time)
5. Write analysis in REPORT.md

Commands:
    python -m student_implementations
"""

"""
STUDENT 2 - LSTM WITH ATTENTION MECHANISM:
TODO in base_sequential_models.py:

1. Implement LSTMModel class:
   - Use nn.LSTM instead of nn.RNN/GRU
   - Implement Bahdanau attention mechanism
   - Architecture:
     * LSTM encoder (bidirectional optional)
     * Attention layer over sequence (computes attention weights)
     * Context vector from attention
     * FC classifier
   
2. Suggested attention mechanism:
   ```python
   class BahdanauAttention(nn.Module):
       def __init__(self, hidden_size):
           super().__init__()
           self.query = nn.Linear(hidden_size, hidden_size)
           self.key = nn.Linear(hidden_size, hidden_size)
           self.value = nn.Linear(hidden_size, 1)
       
       def forward(self, lstm_out):  # lstm_out: (batch, seq_len, hidden*2)
           # Compute attention scores
           scores = self.value(torch.tanh(self.query(lstm_out) + self.key(lstm_out)))
           weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
           context = (weights * lstm_out).sum(dim=1)  # (batch, hidden*2)
           return context, weights
   ```

3. Forward pass should return logits and attention weights
4. Test using example_train_lstm() below
5. Visualize which frames get highest attention weights
"""

"""
STUDENT 3 - TRANSFORMER MODEL:
TODO in base_sequential_models.py:

1. Implement TransformerModel class:
   - Use nn.TransformerEncoder with multi-head self-attention
   - Add positional embeddings for frame positions
   - Architecture:
     * Input projection (512 → hidden_size)
     * Positional embeddings
     * TransformerEncoder (num_layers, num_heads)
     * Temporal pooling (mean over sequence)
     * FC classifier
   
2. Optional: Fine-tune Vision Transformer (ViT) from timm library:
   ```python
   import timm
   vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
   # Extract patch embeddings for sequence modeling
   ```

3. Compare custom Transformer vs ViT
4. Analyze attention patterns via attention weights
"""

"""
STUDENT 4 - UNIFIED COMPARISON & ANALYSIS:
TODO: Create comprehensive_comparison.py

1. Train all 4 models (RNN, GRU, LSTM, Transformer)
2. Evaluate on test set with metrics:
   - Accuracy, Precision, Recall, F1
   - Inference time per sample
   - Model size (parameters)
   - Memory usage
3. Create comparison visualizations:
   - Accuracy bar chart
   - Training curves overlay
   - Confusion matrices (4 subplots)
   - Per-class F1 scores
4. Statistical analysis:
   - t-tests for accuracy differences
   - Temporal agreement (do models agree on same prediction across frames?)
5. Write analysis chapter in REPORT.md
6. Recommend best model for deployment
"""

"""
STUDENT 5 - BONUS FEATURES:
Choose 2-3 of:

A) REAL-TIME SEQUENCE PREDICTION:
   - Load video files
   - Apply models frame-by-frame
   - Visualize prediction confidence evolution
   - Show: frame_idx vs class_prediction vs confidence

B) ADVERSARIAL ROBUSTNESS:
   - Add perturbations to specific frames
   - Test: which model is most robust?
   - Metrics: accuracy drop under attack

C) TEMPORAL ANOMALY DETECTION:
   - Use LSTM hidden states as anomaly features
   - Detect "impossible" sign transitions
   - One-class SVM or Isolation Forest

D) MULTI-TASK LEARNING:
   - Joint training: sign_class + sign_change_detection
   - Does auxiliary task improve main task?

E) EXPLAINABILITY (XAI):
   - LIME/SHAP for sequences
   - Which frames influence prediction most?
   - Compare RNN vs Transformer interpretability
"""


# ============================================================================
# COMPARISON FUNCTION (Student 4)
# ============================================================================

def compare_all_models(results_dir: str = "results"):
    """
    Placeholder for Student 4: Compare all 4 models.
    
    Should:
    1. Train all models
    2. Evaluate on test set
    3. Generate comparison plots
    4. Output summary table
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON (Student 4)")
    print("="*60)
    
    # This will be implemented by Student 4
    print("To be implemented by Student 4")
    print("See student_implementations.py for template")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 3: SEQUENTIAL MODELS TRAINING PIPELINE")
    print("="*80)
    
    # You can run individual examples:
    
    # Example 1: Feature extraction (one-time setup)
    example_extract_vgg16_features()
    
    # Example 2: Create data loaders
    train_loader, val_loader, test_loader = example_create_dataloaders()
    
    # Example 3: Train RNN (uncomment to run)
    # rnn_model, trainer, metrics = example_train_rnn()
    
    # Example 4: Train GRU (uncomment to run)
    # gru_model, trainer, metrics = example_train_gru()
    
    # Example 5: Model comparison (uncomment when Student 4 implements)
    # compare_all_models()
    
    print("\n" + "="*80)
    print("STUDENT INSTRUCTIONS")
    print("="*80)
    print("""
1. STUDENT 1: Implement RNN & GRU training script
   - Models already provided in base_sequential_models.py
   - Use example_train_rnn() as template
   - Create rnn_gru_training.py with full experiments

2. STUDENT 2: Implement LSTM with Attention
   - Edit LSTMModel in base_sequential_models.py
   - Add BahdanauAttention mechanism
   - Create lstm_training.py using UnifiedTrainer

3. STUDENT 3: Implement Transformer
   - Edit TransformerModel in base_sequential_models.py
   - Use nn.TransformerEncoder
   - Optionally fine-tune Vision Transformer
   - Create transformer_training.py

4. STUDENT 4: Unified Comparison & Analysis
   - Create comprehensive_comparison.py
   - Train all 4 models
   - Generate comparison plots and statistics

5. STUDENT 5: Bonus Features
   - Choose 2-3 of: video inference, adversarial robustness, 
     temporal anomaly detection, multi-task learning, XAI
   - Create bonus_features.py

All files should save outputs to results/ directory.
    """)
