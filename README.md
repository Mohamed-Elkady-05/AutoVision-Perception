# AutoVision-Perception

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

A comprehensive, three-phase machine learning framework for traffic sign recognition, evolving from classical machine learning to advanced sequential and transformer-based deep learning models. This project implements a complete pipeline for the German Traffic Sign Recognition Benchmark (GTSRB), including temporal modeling, feature caching, and unified training infrastructure.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Deployment](#Deployment)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Phase 1: Classical Machine Learning](#phase-1-classical-machine-learning)
- [Phase 2: Deep Learning - CNNs](#phase-2-deep-learning---cnns)
- [Phase 3: Sequential & Transformer Models](#phase-3-sequential--transformer-models)
- [Configuration](#configuration)
- [Results & Evaluation](#results--evaluation)
- [References](#references)
---

## Project Overview

**AutoVision-Perception** is a graduated project that systematically explores and implements solutions for traffic sign recognition. The journey begins with fundamental machine learning techniques using hand-crafted features and culminates in state-of-the-art sequential models (RNN, GRU, LSTM, Transformer) that leverage temporal context for improved classification.
### Project Evolution

| Phase | Focus | Key Techniques |
| :--- | :--- | :--- |
| **Phase 1** | Classical Machine Learning | HOG (Histogram of Oriented Gradients), Color Histograms, PCA, KNN, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, Ensemble Methods |
| **Phase 2** | Deep Learning - CNNs | Custom CNN Architectures, Transfer Learning (VGG16), Feature Visualization |
| **Phase 3** | Sequential & Transformer Models | Temporal Sequence Modeling, RNN/GRU/LSTM, Transformer Networks, Feature Caching, Attention Mechanisms |

---

## Key Features

### Across All Phases
- **Complete GTSRB Pipeline**: From raw images to final evaluation
- **Modular Codebase**: Reusable components for feature extraction, training, and evaluation
- **Comprehensive Visualization**: Training curves, confusion matrices, feature maps, and sequence predictions
- **Hyperparameter Optimization**: Grid search for optimal model configuration

### Phase 1 Specifics
- **Feature Engineering**: HOG descriptors + color histograms (420-dimensional features)
- **Dimensionality Reduction**: PCA for variance-based feature selection
- **Multiple Classifiers**: KNN, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting
- **Ensemble Learning**: Voting ensemble combining Random Forest, AdaBoost, and Gradient Boosting

### Phase 2 Specifics
- **Custom CNN**: 3 convolutional blocks with max pooling, dropout regularization
- **Transfer Learning**: VGG16 feature extraction and fine-tuning
- **Feature Visualization**: Intermediate activation maps and filter visualizations

### Phase 3 Specifics
- **Efficient Feature Caching**: Precomputed VGG16 features to accelerate sequence model training
- **Pseudo-Sequence Generation**: Creates 10-frame temporal sequences from static GTSRB images
- **Multiple Sequential Architectures**: RNN, GRU, LSTM (with attention), Transformer models
- **Unified Training Pipeline**: Single training loop with advanced optimization for all model types

---

## Deployment
The Gradio app will:
1. Load all 4 trained models from checkpoints/
2. Display performance metrics from training_summary.json
3. Allow model selection and comparison
4. Show architecture details and test set results

Link: [Phase3 DEPLOYMENT](https://huggingface.co/spaces/AhmedSamir1598/AutoVision-PerceptionHF)

### Files Needed for Deployment

#### Essential
- `app.py` - Gradio interface
- `requirements.txt` - Python dependencies
- `src/` - Source code with models and utilities
- `checkpoints/` - Trained model weights (*.pt files)
- `results/training_summary.json` - Performance metrics

#### Optional
- `README_DEPLOYMENT.md` - Documentation
- `scripts/` - Utility scripts
- Training logs and confusion matrices

### Troubleshooting

#### Models Not Loading
- Check checkpoint paths match: `checkpoints/{model_name}/{model_name}_best.pt`
- Ensure model_metadata loads from: `results/{model_name}_metrics.json`

#### Memory Issues
- Transformer is optimized for CPU (220K params)
- If using GPU Space, models will automatically use GPU
- All models fit in <2GB RAM

#### Import Errors
- Ensure `requirements.txt` includes all dependencies
- Run `pip install -r requirements.txt` locally first to test
---

##  Quick Start

### Prerequisites
- **Python 3.8+**
- **CUDA 11.0+** (optional, for GPU acceleration)
### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohamed-Elkady-05/AutoVision-Perception.git
   cd AutoVision-Perception
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the GTSRB dataset**
   - The notebook automatically downloads from Kaggle using `kagglehub`

5. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA {torch.cuda.is_available()}')"
   ```

---

## Project Structure

```
AutoVision-Perception/
│
├── phase_3.ipynb                 # Main Phase 3 Jupyter notebook
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── notebooks/
│   ├── eda_initial_notebook.ipynb      # Exploratory data analysis (Phase 1)
│   └── phase2_cnn_train.ipynb          # Phase 2 CNN training
|   └── phase3.ipynb                    # pahse 3 RNN and sequential models
│
├── reports/                      # Training metrics & curves
│
├── src/                          # Core source code
│   ├── config.py                 # Centralized configuration
│   │
│   ├── detection/                # Detection & feature extraction
│   │   ├── CNN_model.py          # Custom CNN architecture (Phase 2)
│   │   ├── feature_extractor_vgg16.py  # VGG16 feature extraction (Phase 3)
│   │   └── sequence_dataset.py   # Temporal sequence dataset (Phase 3)
│   │
│   ├── models/
│   │   ├── base_sequential_models.py   # RNN, GRU, LSTM, Transformer (Phase 3)
│   │   ├── baselines.py                # KNN & Naive Bayes (Phase 1)
│   │   ├── ensamble.py                 # Random Forest, AdaBoost,    
|   |   |                               Gradient Boosting 
│   │   │                                 Voting Ensemble (Phase 1)
│   │   ├── grid_search.py              # Hyperparameter search (Phase 1)
│   │   ├── unified_trainer.py          # Unified training loop (Phase 3)
│   │   └── rnn_training_smoke.py       # Real-data RNN training entrypoint
│   │
│   ├── preprocessing/
│   │   ├── feature_extraction.py      # HOG + Color Histogram features (Phase 1)
│   │   └── gtsrb_sequence_preprocessing.py  # Pseudo-sequence builder (Phase 3)
│   │
│   └── utils/
│       ├── cnn_feature_visualization.py   # CNN feature maps (Phase 2)
│       ├── visualize_rnn_sequences.py     # Sequence predictions (Phase 3)
│       ├── sequence_xai.py                # Explainability (Phase 3)
│       └── trainer_opt.py                 # Training optimizations
│
├── checkpoints/                  # Saved model weights
├── cache/                        # Cached VGG16 features
├── data/                         # Dataset directory
└── results/                      # Experiment results
```

---
## Phase 1: Classical Machine Learning

### Feature Extraction
- **HOG (Histogram of Oriented Gradients)**: Captures edge and gradient information
- **Color Histograms**: 32 bins per RGB channel (96 total features)
- **Combined Feature Vector**: 420-dimensional features (HOG + color histograms)
### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Variance-based feature reduction
  - 95% variance threshold
  - StandardScaler for feature normalization
### Ensemble Learning
- using 3 models 
	1. Random Forest Classifier  
	2. Histogram Gradient Boosting Classifier
	3. AdaBoost Classifier
### Grid Search for optimal Hyperparameters
- applying grid search on 2 parameters:
	1. neighbors count `[3, 5, 7]`
	2. knn weights `[uniform, distance]`
### Classifiers Evaluated
| Classifier                  | Type              | Best F1 Score |
| :-------------------------- | :---------------- | :------------ |
| K-Nearest Neighbors         | Distance-based    | ~0.964        |
| Naive Bayes (Gaussian)      | Probabilistic     | ~0.665        |
| Random Forest               | Bagging Ensemble  | ~0.664        |
| Voting Ensemble (RF+Ada+GB) | Boosting Ensemble | ~0.856        |

### Hyperparameter Optimization
- **Grid Search** over:
  - KNN: `n_neighbors` (3, 5, 7), `weights` (uniform, distance)
  - Random Forest: `n_estimators` (100, 200), `max_depth` (None, 15, 30)
  - Gradient Boosting: `n_estimators` (50, 100), `learning_rate` (0.1, 0.2)

### Key Findings
- **KNN with PCA (95% variance)** achieved the best performance: ~96% accuracy
- Ensemble methods (Voting) achieved ~89% accuracy with 95% PCA
- Raw pixel features performed poorly; feature extraction is essential

### Code Examples

```python
from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.baselines import BaselineModels

# Extract HOG + color histogram features
fe = FeatureExtractor(resize=(32, 32))
features = fe.extract_from_list(images)

# Train KNN and Naive Bayes
model = BaselineModels(knn_k=5, use_scaler=True, use_pca=False)
model.fit(X_train, y_train)

# Evaluate
knn_metrics = model.evaluate(X_test, y_test, model='knn')
nb_metrics = model.evaluate(X_test, y_test, model='nb')

---

## Phase 2: Deep Learning - CNNs

**Objective**: Learn hierarchical features end-to-end using convolutional neural networks.

### Custom CNN Architecture
```
Input: 32x32x3
├── Conv1 (3→32, 3x3) → ReLU → MaxPool(2x2)
├── Conv2 (32→64, 3x3) → ReLU → MaxPool(2x2)  
├── Conv3 (64→128, 3x3) → ReLU → MaxPool(2x2)
├── Flatten → FC1 (128*4*4 → 512) → ReLU
├── Dropout(0.5)
└── FC2 (512 → 43) → Output
```

### Transfer Learning
- **VGG16** pretrained on ImageNet
- Extract 512-dimensional features from layer_30
- Features cached for efficient reuse in Phase 3

### Feature Visualization
- **Activation Maps**: Visualize intermediate conv layer outputs
- **Filter Visualization**: Show learned kernel patterns
- **Confidence Bar Plots**: Top-k class predictions with probabilities

### Code Examples

```python
from src.detection.CNN_model import TrafficSignCNN
from src.utils.cnn_feature_visualization import CNNFeatureMapVisualizer

# Create model
model = TrafficSignCNN(num_classes=43)

# Visualize feature maps
visualizer = CNNFeatureMapVisualizer(model, device='cuda')
visualizer.run_from_test_loader(test_loader, num_samples=5, output_dir='viz/')
```

---

## Phase 2: Convolution Neural Networks

### 1. Custom CNN Architecture
A custom CNN model (`TrafficSignCNN`) was implemented with the following architecture:
- Multiple convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Fully connected layers for classification
- Output layer with 43 classes (one for each traffic sign type)

### 2. Data Preparation & Augmentation
- **Training Transforms:**
    - Resizing to 32×32 pixels
    - Random rotation (±15 degrees)
    - Color jitter for brightness variation
    - Normalization to standardize pixel values
- **Dataset Split:** 80% training, 20% validation (31,367 training images, 7,842 validation images)

### 3. Training Configuration
- **Optimizer:** Adam with learning rate 0.001 and L2 regularization (weight decay 1e-4)
- **Loss Function:** Cross-entropy loss
- **Epochs:** 10
- **Batch Size:** 64
- **Hardware:** GPU acceleration (CUDA)

### Performance Results
The custom CNN achieved **99.17% validation accuracy** after 10 epochs, with training loss decreasing from 1.98 to 0.038.

### Transfer Learning Experiments
Two pre-trained models were fine-tuned using a two-phase training approach:
#### Methodology
1. **Warm-up Phase (5 epochs):** Only the classification head is trained while the base model remains frozen
2. **Fine-tuning Phase (10 epochs):** The entire network is unfrozen and trained with a very low learning rate (1e-5)

#### Models Evaluated

##### MobileNetV2
- **Baseline Accuracy (after warm-up):** 46.33%
- **Final Accuracy (after fine-tuning):** 64.05%
- **Characteristics:** Lightweight model, suitable for mobile/edge deployment
##### VGG16
- **Baseline Accuracy (after warm-up):** 71.60%
- **Final Accuracy (after fine-tuning):** 98.80%
- **Characteristics:** Deeper architecture, higher accuracy, computationally intensive

#### Key Insights
- Fine-tuning significantly improved both models (MobileNetV2: +17.72%, VGG16: +27.20%)
- VGG16 achieved accuracy comparable to the custom CNN (98.80% vs 99.17%)
- Transfer learning requires significantly less training time than training from scratch
### Optimizer Comparison
Four optimizers were compared using the custom CNN architecture for 5 epochs each:

| Optimizer   | Final Accuracy | Min Loss |
| ----------- | -------------- | -------- |
| **RMSProp** | 99.08%         | 0.0298   |
| **Adam**    | 98.93%         | 0.0363   |
| **AdaGrad** | 46.39%         | 1.7040   |
| **SGD**     | 36.70%         | 2.0165   |

#### Observations
- **Adam and RMSProp** performed exceptionally well, both achieving >98% accuracy
- **RMSProp** slightly outperformed Adam (99.08% vs 98.93%)
- **AdaGrad and SGD** performed poorly with default learning rates, indicating the need for careful hyperparameter tuning
### Autoencoder for Image Denoising
A convolutional autoencoder was implemented to demonstrate noise reduction capabilities:
#### Architecture
- **Encoder:** Three convolutional layers with ReLU activation
- **Decoder:** Three transposed convolutional layers with Sigmoid activation (output range `[0,1]`)

#### Training Configuration
- **Noise Type:** Gaussian noise (mean=0, std=0.1)
- **Loss Function:** MSE (Mean Squared Error)
- **Epochs:** 5
- **Optimizer:** Adam `(lr=0.001)`

#### Results
- Successfully learned to reconstruct clean images from noisy inputs 
- MSE loss decreased from 0.0195 to 0.0041 over 5 epochs
- Visualizations demonstrate effective noise removal while preserving traffic sign features

### Key Findings & Conclusions
1. **Custom CNN Performance:** A well-designed CNN can achieve >99% accuracy on traffic sign classification with proper training.
2. **Transfer Learning Effectiveness:** Fine-tuning pre-trained models (especially VGG16) achieves near-state-of-the-art performance with less training time.
3. **Optimizer Selection:** Adam and RMSProp are excellent default choices for this task, outperforming SGD and AdaGrad significantly.
4. **Data Augmentation Benefits:** Random rotation and color jitter improved model robustness.
5. **Autoencoder Viability:** Convolutional autoencoders can effectively denoise traffic sign images, which could improve classification in real-world scenarios with noisy input.
## Phase 3: Sequential & Transformer Models

### Pipeline Steps

#### 1. Pseudo-Sequence Generation
- Treat GTSRB as ordered image corpus (sorted within each class)
- Create 10-frame sequences with stride 2
- Each sequence saved as `.npz` with metadata (class label, frame indices, timestamps)
- **Leakage-safe grouped splits**: Ensures no frame overlap between train/val/test

#### 2. Feature Extraction & Caching
- Extract 512-dimensional VGG16 features for all frames
- Cache features to `cache/vgg16_sequence_features/` for rapid training
- Precompute features once, reuse across all sequential models

#### 3. Sequential Models (Unified Interface)

| Model           | Architecture           | Key Features                                                        |
| :-------------- | :--------------------- | :------------------------------------------------------------------ |
| **RNN**         | 2-layer, bidirectional | Vanilla RNN with mean pooling                                       |
| **GRU**         | 2-layer, bidirectional | Gated recurrent units, faster training                              |
| **LSTM**        | 2-layer, bidirectional | Long short-term memory + additive attention                         |
| **Transformer** | 1-layer encoder        | Multi-head self-attention (4 heads), sinusoidal positional encoding |

#### 4. Unified Training
- Single trainer compatible with all models
- Multiple optimizers: Adam, SGD, AdamW
- Learning rate schedulers: Step, Cosine, ReduceLROnPlateau
- Early stopping with configurable patience
- Best model checkpointing
### Running Phase 3

```bash
# 1. Generate pseudo-sequences and cache VGG16 features
python -m src.preprocessing.gtsrb_sequence_preprocessing

# 2. Train models (each for 50 epochs)
python -m src.models.rnn_training_smoke
python -m src.models.transformer_training_smoke

# 3. Run complete experiment (train all 4 models)
python src/models/ModelsP3_experiment.py

# 4. Visualize predictions with XAI
python -m src.utils.visualize_rnn_sequences
```

---

## Configuration

All pipeline parameters centralized in `src/config.py`:

```python
from src.config import TrainingConfig, SequentialModelConfig

# Training settings
train_cfg = TrainingConfig()
train_cfg.NUM_EPOCHS = 50
train_cfg.BATCH_SIZE = 32
train_cfg.LEARNING_RATE = 0.001
train_cfg.OPTIMIZER_TYPE = 'adam'      # 'adam', 'sgd', 'adamw'
train_cfg.SCHEDULER_TYPE = 'cosine'    # 'step', 'cosine', 'reduce_on_plateau'

# Sequential model settings
model_cfg = SequentialModelConfig()
model_cfg.SEQUENCE_LENGTH = 10
model_cfg.INPUT_SIZE = 512
model_cfg.HIDDEN_SIZE = 256
model_cfg.NUM_LAYERS = 2
model_cfg.DROPOUT = 0.3
model_cfg.BIDIRECTIONAL = True
```

---

## Results & Evaluation

### Phase 1 Results

| Model                      | Accuracy | Precision | Recall | F1 Score |
| :------------------------- | :------: | :-------: | :----: | :------: |
| KNN                        |  88.0%   |   89.5%   | 87.8%  |  88.4%   |
| Naive Bayes                |  44.2%   |   63.6%   | 46.5%  |  50.8%   |
| Random Forest (Bagging)    |  73.9%   |   87.4%   | 61.4%  |  66.4%   |
| Voting Ensemble (Boosting) |  89.2%   |   90.8%   | 82.1%  |  85.6%   |

### Phase 2 Results
...
### Phase 3 Results

| Model       | Accuracy | Precision | Recall | F1 score |
| ----------- | :------: | :-------: | :----: | :------: |
| RNN         |  84.5%   |   79.5%   | 79.5%  |  79.5%   |
| GRU         |  83.1%   |   78.8%   | 78.3%  |  77.0%   |
| LSTM        |  83.1%   |   77.7%   | 79.5%  |  77.1%   |
| Transformer |  85.7%   |   81.1%   | 81.1%  |  75.0%   |

---

## GPU Acceleration

The framework automatically detects CUDA-capable GPUs:

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = UnifiedTrainer(model, train_loader, val_loader, device=device)
```
---
## References

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
---

**Last Updated:** May 2026  
**Current Version:** 3.0 - Phase 3 Implementation
