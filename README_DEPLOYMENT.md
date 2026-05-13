# 🚦 AutoVision-Perception: GTSRB Traffic Sign Classification

A state-of-the-art deep learning project comparing **4 sequence models** (RNN, GRU, LSTM, Transformer) for traffic sign recognition using VGG16 feature extraction and pseudo-sequence modeling.

## 📊 Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| RNN | 84.53% | 79.59% | 80.58% | 78.64% | 930K |
| GRU | 83.18% | 78.89% | 78.30% | 77.01% | 2.5M |
| LSTM | 83.18% | 77.77% | 79.54% | 77.14% | 3.6M |
| **Transformer** | **85.75%** | **81.12%** | **81.18%** | **79.49%** | **220K** |

🏆 **Best Model**: Transformer with 85.75% test accuracy

---

## 🏗️ Architecture

### Data Pipeline
- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Feature Extraction**: VGG16 (512-dimensional features)
- **Sequences**: 10-frame pseudo-sequences with 2-frame stride
- **Total Sequences**: 3,920 (2,654 train | 529 val | 737 test)
- **Classes**: 43 traffic sign categories

### Models

1. **RNN** - Vanilla Recurrent Neural Network
   - Bidirectional processing
   - 930K parameters
   - Best for simple sequential patterns

2. **GRU** - Gated Recurrent Unit
   - Gating mechanisms for sequence modeling
   - 2.5M parameters
   - Faster than LSTM with comparable performance

3. **LSTM** - Long Short-Term Memory
   - Attention mechanism (additive attention)
   - 3.6M parameters
   - Handles long-term dependencies

4. **Transformer** (🏆 Best)
   - Self-attention architecture
   - Only 220K parameters (most efficient)
   - Parallel sequence processing
   - Optimized for CPU inference

---

## 📈 Key Findings

### Test vs Validation Set Differences
- **Validation Set** (529 sequences): Used during training for early stopping and checkpoint selection
- **Test Set** (737 sequences): Held out completely for unbiased final evaluation
- **Why Different**: Test and validation sets have different class distributions; model was optimized for validation, not test

### Why Transformer is Best
1. **Parameter Efficiency**: 220K params vs 3.6M (LSTM)
2. **Parallel Processing**: Self-attention processes all frames simultaneously
3. **Fast Inference**: Optimized for both CPU and GPU
4. **Strong Generalization**: 85.75% test accuracy with early stopping at epoch 13

---

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone https://huggingface.co/spaces/AhmedSamir1598/AutoVision-Perception
cd AutoVision-Perception

# Install dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py

# Open browser to http://localhost:7860
```

### Training from Scratch
```bash
# Precompute VGG16 features
python -m src.preprocessing.gtsrb_sequence_preprocessing

# Train all models using unified pipeline
python -m src.models.ModelsP3_experiment

# Generate confusion matrices
python scripts/plot_confusion_matrices.py
```

---

## 📁 Project Structure

```
AutoVision-Perception/
├── app.py                          # Gradio interface for HF Spaces
├── requirements.txt                # Python dependencies
├── src/
│   ├── config.py                  # Configuration constants
│   ├── models/
│   │   ├── base_sequential_models.py  # RNN, GRU, LSTM, Transformer
│   │   ├── unified_trainer.py          # Training loop
│   │   ├── ModelsP3_experiment.py      # Unified training script
│   │   ├── sequence_model_runner.py    # Shared runner utilities
│   │   └── *_training_smoke.py         # Individual model trainers
│   ├── detection/
│   │   ├── sequence_dataset.py         # Data loading
│   │   └── feature_extractor_vgg16.py  # VGG16 feature extraction
│   ├── preprocessing/
│   │   └── gtsrb_sequence_preprocessing.py  # Sequence building
│   └── utils/
│       └── sequence_xai.py             # Explainability helpers
├── results/
│   ├── *_metrics.json              # Test metrics per model
│   ├── *_confusion_matrix.png      # Confusion matrix visualizations
│   ├── *_training_curves.png       # Training/validation curves
│   └── training_summary.json       # Consolidated results
├── checkpoints/
│   └── {model_name}/*_best.pt      # Trained model weights
└── scripts/
    └── plot_confusion_matrices.py  # Visualization script
```

---

## 🔧 Technical Details

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Batch Size**: 32
- **Max Epochs**: 30
- **Early Stopping**: 10 epochs patience
- **Scheduler**: Cosine annealing
- **Device**: CPU (models CPU-optimized)

### Sequence Generation
- **Sequence Length**: 10 frames
- **Frame Stride**: 2 (reduces overlap)
- **Grouping Strategy**: Non-overlapping groups by video source + segment ID
- **Prevents Data Leakage**: No sequence appears in multiple splits

### Feature Extraction
- **Model**: VGG16 (pre-trained on ImageNet)
- **Layer**: activations_layer_4
- **Dimension**: 512
- **Normalization**: Standard scaling
- **Caching**: Pre-computed for speed

---

## 📊 Metrics Explanation

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positives, what % are actually correct
- **Recall**: Of actual positives, what % were found
- **F1 Score**: Harmonic mean of precision & recall

---

## 🤝 Contributing

To extend this project:

1. Add new sequence models to `src/models/base_sequential_models.py`
2. Create training script in `src/models/{model_name}_training_smoke.py`
3. Update `ModelsP3_experiment.py` main loop
4. Run comparisons and update results

---

## 📝 References

- **GTSRB Dataset**: [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
- **VGG16**: Simonyan & Zisserman, 2014
- **Transformers**: Vaswani et al., 2017
- **PyTorch**: [Official Documentation](https://pytorch.org)
- **Gradio**: [Spaces Deployment Guide](https://huggingface.co/docs/hub/spaces)

---

## 📄 License

This project is part of academic coursework for supervised learning.

---

## 🙋 Support

For questions or issues:
- Check the confusion matrices in `results/`
- Review training logs in `results/*_training_curves.png`
- Inspect metrics in `results/training_summary.json`
