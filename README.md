# AutoVision-Perception: Phase 3 Sequential & Transformer Models

A comprehensive machine learning framework for traffic sign recognition using sequential and transformer-based deep learning models. This project implements temporal modeling pipelines with precomputed CNN features, advanced sequence architectures (RNN, GRU, LSTM, Transformer), and unified training infrastructure.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Results & Evaluation](#results--evaluation)

## Project Overview

**AutoVision-Perception** extends the AutoVision project with Phase 3 implementation, introducing temporal modeling for improved traffic sign recognition. The framework now supports real pseudo-sequences built from the extracted GTSRB archive, cached VGG16 sequence features, RNN training on the real cache, and visual inspection of saved sequence frames.

**Key Features:**
- Efficient feature extraction and caching with VGG16
- Temporal sequence generation (10-frame sequences)
- Real-data pseudo-sequence generation from sorted class folders
- Multiple sequential architectures (RNN, GRU, LSTM, Transformer)
- Unified training pipeline with advanced optimization
- Comprehensive evaluation metrics and visualization
- GPU-accelerated training with PyTorch

## Architecture

```
AUTOVISION PIPELINE - PHASE 3
├─ Feature Extraction (VGG16)
│  └─ 512-dimensional feature vectors
├─ Sequence Generation (10-frame temporal)
│  └─ (batch, seq_len=10, 512) tensors
├─ Sequential Models
│  ├─ RNN - Vanilla Recurrent Neural Network
│  ├─ GRU - Gated Recurrent Unit
│  ├─ LSTM - Long Short-Term Memory + Attention
│  └─ Transformer - Multi-head Transformer Architecture
├─ Unified Training
│  └─ Single training loop for all models
├─ Evaluation & Comparison
│  └─ Metrics, visualizations, statistical analysis
└─ Extensions
   ├─ Real-time inference
   ├─ Adversarial robustness testing
   ├─ Anomaly detection
   └─ Interpretability analysis
```

## Real Sequence Workflow

The preprocessing pipeline treats GTSRB as an ordered image corpus rather than a native video dataset.

1. Read images from `archive/Train/<class>/`.
2. Sort images inside each class folder.
3. Slice windows of length 10 with stride 2.
4. Save each window as a `.npz` file with `frames` and JSON metadata.
5. Precompute VGG16 features for each saved sequence and cache them under `cache/vgg16_sequence_features/`.

The metadata preserves the class label, frame indices, timestamps, and folder name so each saved sequence can be traced back to the exact image block it came from.

## Prerequisites

- **Python 3.8+**
- **CUDA 11.0+** (optional, for GPU acceleration)
- **GPU Memory**: 4GB+ recommended (8GB+ for large batch sizes)

## Installation

### 1. Clone and Navigate to Project Directory

```bash
cd AutoVision-Perception
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `scikit-image` - Image processing
- `Pillow` - Image handling
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
AutoVision-Perception/
├── phase_3.ipynb                 # Main Phase 3 notebook
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/                           # Dataset directory
│   └── [image sequences]
├── notebooks/
│   ├── eda_initial_notebook.ipynb  # Exploratory data analysis
│   └── phase2_cnn_train.ipynb      # Phase 2 CNN training
├── reports/
│   └── [training outputs & metrics]
└── src/
    ├── config.py                   # Centralized configuration
    ├── pipeline_example.py          # Example pipeline
    ├── detection/
    │   ├── CNN_model.py            # CNN architecture
    │   ├── feature_extractor_vgg16.py # VGG16 feature extraction
    │   └── sequence_dataset.py      # Temporal sequence dataset
    ├── models/
    │   ├── base_sequential_models.py  # RNN, GRU base classes
    │   ├── baselines.py              # Baseline models
    │   ├── ensamble.py               # Ensemble methods
    │   ├── grid_search.py            # Hyperparameter search
    │   ├── ModelsP3_experiment.py    # Phase 3 experiments
    │   ├── rnn_training_smoke.py     # Real-data RNN training entrypoint
    │   └── unified_trainer.py        # Unified training loop
    ├── preprocessing/
    │   ├── feature_extraction.py    # Feature extraction utilities
    │   └── gtsrb_sequence_preprocessing.py # Pseudo-sequence builder and feature cache
    └── utils/
        ├── cnn_feature_visualization.py  # Visualization tools
        ├── student_implementations.py    # Student templates
        ├── visualize_rnn_sequences.py    # Real sequence visualization helper
        └── trainer_opt.py                # Training optimization
```

## Quick Start

### Option 1: Run the Main Notebook

```bash
jupyter notebook phase_3.ipynb
```

Execute cells sequentially to:
1. Load and configure the pipeline
2. Extract VGG16 features
3. Create sequence dataloaders
4. Train models (RNN, GRU)
5. Evaluate and compare results

### Option 2: Run Python Scripts

#### Extract VGG16 Features

```bash
python -c "
from src.detection.feature_extractor_vgg16 import VGG16FeatureExtractor
from src.config import FeatureExtractorConfig

config = FeatureExtractorConfig()
extractor = VGG16FeatureExtractor(config)
features = extractor.extract_and_cache(
    image_paths=['path/to/images/'],
    cache_key='my_dataset'
)
print(f'Extracted features shape: {features.shape}')
"
```

#### Create Sequence DataLoaders

```bash
python -c "
from src.detection.sequence_dataset import create_sequence_dataloaders

train_loader, val_loader, test_loader = create_sequence_dataloaders(
    features_dir='./cache/vgg16_sequence_features',
    batch_size=32
)
print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')
print(f'Test batches: {len(test_loader)}')
"
```

#### Build the real pseudo-sequences and cached features

```bash
python -m src.preprocessing.gtsrb_sequence_preprocessing
```

#### Train the RNN on the real cache

```bash
python -m src.models.rnn_training_smoke
```

#### Train the GRU model

```bash
python -m src.models.gru_training_smoke
```

#### Train the LSTM model

```bash
python -m src.models.lstm_training_smoke
```

#### Train the Transformer model

```bash
python -m src.models.transformer_training_smoke
```

#### Visualize saved sequence frames with RNN predictions

```bash
python -m src.utils.visualize_rnn_sequences
```

#### Launch the Hugging Face Space app locally

```bash
python app.py
```

The app accepts either a cached sequence `.npz` file or a single image fallback. It returns the predicted class, confidence, and a saved explanation plot.

#### Explainability outputs

Each training run now saves a matching XAI artifact in `results/`, for example:

- `results/RNN_xai.png`
- `results/GRU_xai.png`
- `results/LSTM_xai.png`
- `results/Transformer_xai.png`

#### Train a Model

```bash
python -c "
from src.models.base_sequential_models import RNNModel, create_model
from src.models.unified_trainer import UnifiedTrainer
from src.config import TrainingConfig
import torch

# Create model
model = RNNModel(
    input_size=512,
    hidden_size=256,
    num_layers=2,
    output_size=43,
    dropout=0.3
)

# Create trainer
config = TrainingConfig()
trainer = UnifiedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train
trainer.train(num_epochs=50)

# Evaluate
metrics = trainer.evaluate(test_loader)
print(f'Test Accuracy: {metrics[\"accuracy\"]:.4f}')
"
```

## Core Components

### 1. Configuration System (`src/config.py`)

Centralized configuration management for all pipeline components.

**Key Classes:**
- `DatasetConfig` - Sequence parameters, normalization settings
- `FeatureExtractorConfig` - VGG16 settings, cache paths
- `SequentialModelConfig` - Model architecture specifications
- `TrainingConfig` - Optimization, learning rates, schedulers
- `EvaluationConfig` - Metrics and visualization settings

**Usage:**
```python
from src.config import TrainingConfig

config = TrainingConfig()
config.learning_rate = 0.001
config.batch_size = 32
config.num_epochs = 50
```

### 2. Feature Extraction (`src/detection/feature_extractor_vgg16.py`)

Extracts and caches deep CNN features using pretrained VGG16.

**Key Methods:**
- `extract_single(image)` → (512,) feature vector
- `extract_sequence(images)` → (seq_len, 512) features
- `extract_batch(image_paths)` → (num_images, 512) features
- `extract_and_cache(image_paths, cache_key)` → cached features

### 3. Sequence Dataset (`src/detection/sequence_dataset.py`)

Loads precomputed sequence features with metadata and synthetic fallback support.

### 4. Pseudo-sequence Preprocessing (`src/preprocessing/gtsrb_sequence_preprocessing.py`)

Creates pseudo-sequences from the archived GTSRB class folders and precomputes the VGG16 sequence cache.

### 5. RNN Smoke Trainer (`src/models/rnn_training_smoke.py`)

Trains the RNN for 50 epochs on the real cache and writes the checkpoint, metrics, curves, and confusion matrix.

### 6. Sequence Visualizer (`src/utils/visualize_rnn_sequences.py`)

Renders saved frames from the real sequence cache and overlays the RNN predictions.

Generated artifacts are ignored by Git:

- `archive/`
- `archive.zip`
- `data/`
- `cache/`
- `checkpoints/`
- `results/`

**Key Features:**
- Temporal sequence generation (10-frame default)
- Stratified train/val/test split
- Temporal augmentation (flip, noise, frame dropping)
- Synthetic data generation for testing

### 4. Sequential Models (`src/models/base_sequential_models.py`)

Base classes and implementations for sequential architectures.

**Available Models:**
- `RNNModel` - Vanilla RNN
- `GRUModel` - Gated Recurrent Unit
- `LSTMModel` - LSTM with attention (template)
- `TransformerModel` - Multi-head Transformer (template)

**Usage:**
```python
from src.models.base_sequential_models import create_model

model = create_model(
    model_name='rnn',
    config_dict={'input_size': 512, 'hidden_size': 256},
    device='cuda'
)
```

### 5. Unified Trainer (`src/models/unified_trainer.py`)

Single training loop compatible with all model architectures.

**Features:**
- Multiple optimizers (Adam, SGD, AdamW)
- Learning rate schedulers (step, cosine, reduce_on_plateau)
- Early stopping with patience
- Best model checkpointing
- Comprehensive metrics computation
- Training visualization

## Configuration

Edit `src/config.py` to customize pipeline behavior:

```python
# Dataset configuration
dataset_config = DatasetConfig()
dataset_config.sequence_length = 10  # Number of frames per sequence
dataset_config.train_ratio = 0.7
dataset_config.val_ratio = 0.15

# Training configuration
train_config = TrainingConfig()
train_config.num_epochs = 50
train_config.batch_size = 32
train_config.learning_rate = 0.001
train_config.optimizer_type = 'adam'  # 'adam', 'sgd', or 'adamw'
train_config.scheduler_type = 'cosine'  # 'step', 'cosine', or 'reduce_on_plateau'

# Model configuration
model_config = SequentialModelConfig()
model_config.hidden_size = 256
model_config.num_layers = 2
model_config.dropout = 0.3
```

## Data Augmentation

Data augmentation is applied **only during training** to improve model robustness and prevent overfitting. Augmentation is implemented in `src/detection/sequence_dataset.py` and operates on temporal sequences of features.

### Augmentation Techniques

1. **Temporal Frame Reversal** (30% probability)
   - Reverses the order of frames within each sequence
   - Exposes the model to backward temporal patterns
   - Regularizes the model to capture bidirectional dependencies
   - Implementation: `sequence = sequence[::-1]`

2. **Gaussian Noise Injection** (20% probability)
   - Adds small random Gaussian noise to feature values
   - Noise scale: `std_dev = 0.05` of feature magnitude
   - Improves robustness to minor feature perturbations
   - Implementation: `sequence += np.random.randn(*sequence.shape) * 0.05`

### Configuration

Augmentation settings are controlled in `src/config.py`:

```python
# Enable/disable augmentation per split
dataset_config.augment_train = True    # Apply to training split
dataset_config.augment_val = False     # No augmentation on validation
dataset_config.augment_test = False    # No augmentation on test

# Augmentation probabilities (in sequence_dataset.py)
TEMPORAL_FLIP_PROB = 0.3
NOISE_INJECTION_PROB = 0.2
NOISE_SCALE = 0.05
```

### Benefits

- **Temporal Regularization**: Prevents the model from learning overly-specific frame orderings
- **Feature Robustness**: Noise injection simulates feature extraction variance
- **Reduced Overfitting**: Increases effective training data diversity
- **No Information Loss**: Both augmentations are mathematically invertible

---

## Data Processing Pipeline

This section describes the complete workflow from downloading the GTSRB dataset from Kaggle to training on cached features.

### Step 1: Download from Kaggle

Visit [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and download the dataset.

**Expected structure after extraction:**
```
archive/
├── Train/
│   ├── 0/
│   │   ├── 00000.ppm
│   │   ├── 00001.ppm
│   │   └── ...
│   ├── 1/
│   │   ├── 00000.ppm
│   │   └── ...
│   └── ... (42 total classes)
└── Test/
    └── (optional, not used in Phase 3)
```

The dataset contains **43 traffic sign classes** (0-42) with images in `.ppm` format.

### Step 2: Create Pseudo-Sequences from Archive

The preprocessing pipeline (`src/preprocessing/gtsrb_sequence_preprocessing.py`) treats the GTSRB archive as an **ordered image corpus** rather than a native video dataset.

**Process:**
1. Read all images from each class folder: `archive/Train/<class_id>/`
2. Sort images alphabetically within each class
3. Create overlapping temporal windows using a **sliding window**:
   - **Window size (sequence length)**: 10 frames
   - **Stride**: 2 frames (produces overlapping sequences)
   - Creates pseudo-video sequences from static images
4. Save each sequence as a `.npz` file with metadata

**Example:**
```
Class 0 images: [00000.ppm, 00001.ppm, 00002.ppm, ..., 00500.ppm]

Sequences created (with stride=2):
- Seq 0: frames [00000-00009]
- Seq 1: frames [00002-00011]
- Seq 2: frames [00004-00013]
- ... (continues every 2 frames)
```

**Output:** `data/preprocessed_sequences/{split}/{class}/*.npz`
- Each `.npz` contains:
  - `frames`: stacked image array shape (10, H, W, 3)
  - `metadata`: JSON with sequence ID, start frame, class label, timestamps

### Step 3: Precompute VGG16 Features

Feature extraction is performed once and cached to avoid recomputation (`SequenceFeaturePrecomputer` in `gtsrb_sequence_preprocessing.py`).

**Process:**
1. Load each sequence's 10 frames from `.npz`
2. Extract features using **pretrained VGG16** (before classification layer)
3. Output: **512-dimensional feature vectors** per frame
4. Cache features as compressed `.npz` files

**Output:** `cache/vgg16_sequence_features/{split}/{class}/*.npz`
- Each cached file contains:
  - `features`: shape (10, 512) - sequence of feature vectors
  - `metadata`: preserved from preprocessing step

**Why VGG16?**
- Pretrained on ImageNet: captures universal image patterns
- 512-dimensional output: rich representation, computationally efficient
- Proven performance on traffic sign recognition tasks
- Transfer learning reduces need for large training data

### Step 4: Generate Leakage-Safe Train/Val/Test Splits

The `regenerate_grouped_feature_splits()` function creates **non-overlapping, leakage-free splits**:

**Leakage Prevention Strategy:**
1. **Filter Non-Overlapping Windows**
   - Keep only sequences where `start_frame % sequence_length == 0`
   - Ensures no frame appears in multiple sequences across splits
   
2. **Group by Source Segment**
   - Related sequences from the same image block stay together
   - `group_key = f"{source}::segment_{start_frame // (seq_len * group_size)}"`
   
3. **Stratified Split by Group**
   - Split at the **group level**, not individual sequence level
   - Prevents frames from one group appearing in both train and test

**Result:**
- **Train**: 2,654 sequences (70%)
- **Validation**: 529 sequences (15%)
- **Test**: 737 sequences (15%)
- **Total**: 3,920 non-overlapping sequences across 43 classes

### Step 5: Load and Train on Cached Features

The `SequenceDataset` loads cached features efficiently:

```python
from src.detection.sequence_dataset import create_sequence_dataloaders

# Automatically detects train/val/test split folders
train_loader, val_loader, test_loader = create_sequence_dataloaders(
    features_dir="./cache/vgg16_sequence_features",
    batch_size=32,
    augment=True  # Enables augmentation for training split
)
```

**Data Flow:**
```
cache/vgg16_sequence_features/
├── train/ → SequenceDataset (apply augmentation)
│           → DataLoader (shuffled batches)
├── val/   → SequenceDataset (no augmentation)
│           → DataLoader (ordered batches)
└── test/  → SequenceDataset (no augmentation)
            → DataLoader (ordered batches)
```

### Complete Pipeline Command

To run the entire preprocessing and feature caching pipeline:

```bash
python -c "
from src.preprocessing.gtsrb_sequence_preprocessing import (
    SequencePreprocessor,
    SequenceFeaturePrecomputer,
    regenerate_grouped_feature_splits
)
from pathlib import Path

# Step 1: Create pseudo-sequences from archive
preprocessor = SequencePreprocessor()
for split in ['train', 'val', 'test']:
    preprocessor.preprocess_all(split=split)

# Step 2: Precompute VGG16 features
precomputer = SequenceFeaturePrecomputer(device='cuda')
precomputer.precompute_all_splits(splits=['train', 'val', 'test'])

# Step 3: Generate leakage-safe splits
regenerate_grouped_feature_splits()

print('✓ Data processing complete!')
print('✓ Cached features ready in: cache/vgg16_sequence_features/')
"
```

Or run directly:
```bash
python -m src.preprocessing.gtsrb_sequence_preprocessing
```

---

## Running Experiments

### Run Complete Pipeline

```bash
python src/utils/student_implementations.py
```

This executes:
1. VGG16 feature extraction
2. Sequence dataset creation
3. Model training (RNN, GRU)
4. Evaluation and comparison

### Run Hyperparameter Grid Search

```bash
python src/models/grid_search.py
```

Searches over parameter ranges defined in `GridSearchConfig`.

### Run Model Ensemble

```bash
python -c "
from src.models.ensamble import EnsembleModel
from src.models.base_sequential_models import RNNModel, GRUModel

# Create ensemble
rnn = RNNModel(512, 256, 2, 43)
gru = GRUModel(512, 256, 2, 43)
ensemble = EnsembleModel(models=[rnn, gru], weights=[0.5, 0.5])

# Use like a regular model
output = ensemble(batch_tensor)
"
```

## Results & Evaluation

### Generate Training Curves

```python
trainer.save_training_curves(output_dir='reports/')
# Outputs: training_curves.png
```

### Export Metrics

```python
metrics = trainer.evaluate(test_loader)
trainer.save_metrics_json(metrics, output_dir='reports/')
# Outputs: metrics.json with accuracy, precision, recall, F1, confusion matrix
```

### Metrics Included

- **Accuracy** - Overall classification accuracy
- **Precision** - Per-class precision
- **Recall** - Per-class recall
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Prediction breakdown
- **Training Curves** - Loss and accuracy over epochs

## GPU Acceleration

The framework automatically detects CUDA availability. To force CPU:

```python
import torch
device = 'cpu'  # or 'cuda' for GPU

trainer = UnifiedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device
)
```

Check GPU status:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Current Device: {torch.cuda.current_device()}')"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce `batch_size` in config or use CPU |
| `Feature cache not found` | Run feature extraction first using `VGG16FeatureExtractor` |
| `No data found` | Ensure images are in `data/` directory with correct structure |
| Slow training on CPU | Enable CUDA with `device='cuda'` |

## References

- [PyTorch Documentation](https://pytorch.org/docs)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Traffic Sign Recognition Dataset (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)

---

**Last Updated:** May 2026  
**Version:** 3.0 - Phase 3 Implementation