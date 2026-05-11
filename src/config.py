"""
Phase 3 Configuration - Sequential Models (RNN, GRU, LSTM, Transformer)
Centralized config for dataset, training, and model hyperparameters.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "sequence_data"
CACHE_DIR = PROJECT_ROOT / "cache" / "vgg16_features"
MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for directory in [DATA_DIR, CACHE_DIR, MODEL_CHECKPOINT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
class DatasetConfig:
    """Configuration for sequence dataset."""
    
    # Sequence parameters
    SEQUENCE_LENGTH = 10  # frames per sequence
    FRAME_STRIDE = 2      # step between frames (skip frames to create temporal diversity)
    IMAGE_SIZE = (224, 224)  # VGG16 expects 224x224
    NUM_CLASSES = 43      # GTSRB traffic sign classes
    
    # Dataset split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Data augmentation (during sequence construction)
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.5
    
    # Preprocessing
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean (for VGG16)
    NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std


# ============================================================================
# FEATURE EXTRACTION CONFIGURATION (VGG16)
# ============================================================================
class FeatureExtractorConfig:
    """Configuration for CNN feature extraction."""
    
    # Model
    FEATURE_EXTRACTOR_MODEL = "vgg16"  # pretrained on ImageNet
    FEATURE_LAYER = "features_layer_30"  # VGG16 layer for feature extraction
    FEATURE_DIM = 512  # Output feature dimension from VGG16
    
    # Processing
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = "cuda"  # or "cpu"
    
    # Caching
    CACHE_FEATURES = True
    CACHE_DIR = CACHE_DIR


# ============================================================================
# MODEL CONFIGURATION - SEQUENTIAL MODELS
# ============================================================================
class SequentialModelConfig:
    """Base configuration for all sequential models."""
    
    # Input/Output
    INPUT_SIZE = 512  # VGG16 feature dimension
    HIDDEN_SIZE = 256  # Hidden state dimension
    NUM_LAYERS = 2
    OUTPUT_SIZE = 43  # GTSRB classes
    SEQUENCE_LENGTH = DatasetConfig.SEQUENCE_LENGTH
    
    # Regularization
    DROPOUT = 0.3
    BIDIRECTIONAL = True  # for RNN/GRU/LSTM
    
    # Attention (for LSTM and Transformer)
    USE_ATTENTION = True
    ATTENTION_HEADS = 4  # for Transformer


class RNNGRUConfig(SequentialModelConfig):
    """Configuration for RNN and GRU models."""
    pass


class LSTMConfig(SequentialModelConfig):
    """Configuration for LSTM with Attention."""
    USE_ATTENTION = True
    ATTENTION_HIDDEN = 128


class TransformerConfig(SequentialModelConfig):
    """Configuration for Transformer model."""
    FFN_DIM = 1024
    NUM_TRANSFORMER_LAYERS = 4
    ATTENTION_HEADS = 8
    DROPOUT = 0.2
    ATTENTION_DROPOUT = 0.1


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
class TrainingConfig:
    """Configuration for model training."""
    
    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Schedulers
    SCHEDULER_TYPE = "cosine"  # "step", "cosine", "reduce_on_plateau"
    SCHEDULER_STEP_SIZE = 10   # for step scheduler
    SCHEDULER_GAMMA = 0.1      # decay factor
    
    # Loss
    LOSS_FUNCTION = "cross_entropy"
    CLASS_WEIGHTS = None  # set if class imbalance
    
    # Device
    DEVICE = "cuda"
    
    # Checkpointing
    SAVE_BEST_ONLY = True
    CHECKPOINT_DIR = MODEL_CHECKPOINT_DIR
    
    # Logging
    LOG_INTERVAL = 100  # batches
    VAL_INTERVAL = 1    # epochs


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
class EvaluationConfig:
    """Configuration for model evaluation and comparison."""
    
    # Metrics
    COMPUTE_METRICS = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
        "per_class_accuracy"
    ]
    
    # Visualization
    PLOT_CONFUSION_MATRIX = True
    PLOT_CLASS_DISTRIBUTION = True
    PLOT_TEMPORAL_AGREEMENT = True  # agreement across sequence frames
    
    # Output
    RESULTS_DIR = RESULTS_DIR


# ============================================================================
# INFERENCE CONFIGURATION (for later, not yet implemented)
# ============================================================================
class InferenceConfig:
    """Configuration for inference on video sequences."""
    
    DEVICE = "cuda"
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    
    # Post-processing
    USE_CONFIDENCE_THRESHOLD = True
    CONFIDENCE_THRESHOLD = 0.8
    
    # Temporal smoothing
    USE_TEMPORAL_SMOOTHING = False
    SMOOTHING_WINDOW = 3


# ============================================================================
# EXPORT ALL CONFIGS
# ============================================================================
def get_all_configs():
    """Returns dictionary of all configuration objects."""
    return {
        "dataset": DatasetConfig,
        "feature_extractor": FeatureExtractorConfig,
        "rnn_gru": RNNGRUConfig,
        "lstm": LSTMConfig,
        "transformer": TransformerConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "inference": InferenceConfig,
    }


if __name__ == "__main__":
    # Quick test
    configs = get_all_configs()
    for name, config in configs.items():
        print(f"\n{name.upper()} CONFIG:")
        for attr in dir(config):
            if not attr.startswith("_"):
                value = getattr(config, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
