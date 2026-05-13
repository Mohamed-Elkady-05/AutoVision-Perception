# 🚀 Deployment Instructions for Hugging Face Spaces

## Step 1: Clone the HF Space Repository

First, you'll need to clone the existing Space repository. When prompted for a password, use a GitHub Personal Access Token or HF token with write permissions.

```bash
# Generate HF token from: https://huggingface.co/settings/tokens
git clone https://huggingface.co/spaces/AhmedSamir1598/AutoVision-Perception
cd AutoVision-Perception
```

## Step 2: Install HF CLI (Windows PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Then authenticate:
```powershell
huggingface-cli login
```

## Step 3: Copy Project Files to Space

Copy the following files from this local repo to the cloned Space:

**Critical Files:**
```
app.py                                    → Copy to Space root
requirements.txt                          → Copy to Space root
README_DEPLOYMENT.md                      → Copy as README.md
src/models/base_sequential_models.py     → Copy entire src/ folder
src/models/unified_trainer.py
src/detection/sequence_dataset.py
src/config.py
checkpoints/rnn/RNN_best.pt              → Copy checkpoints/ folder
checkpoints/gru/GRU_best.pt
checkpoints/lstm/LSTM_best.pt
checkpoints/transformer/Transformer_best.pt
results/training_summary.json            → Copy results/ folder
results/*_metrics.json
```

## Step 4: Commit and Push to HF Spaces

```bash
git add .
git commit -m "Add AutoVision-Perception models and Gradio app"
git push
```

## Step 5: Monitor Deployment

Visit your Space: `https://huggingface.co/spaces/AhmedSamir1598/AutoVision-Perception`

The Gradio app will:
1. Load all 4 trained models from checkpoints/
2. Display performance metrics from training_summary.json
3. Allow model selection and comparison
4. Show architecture details and test set results

---

## Files Needed for Deployment

### Essential
- `app.py` - Gradio interface
- `requirements.txt` - Python dependencies
- `src/` - Source code with models and utilities
- `checkpoints/` - Trained model weights (*.pt files)
- `results/training_summary.json` - Performance metrics

### Optional
- `README_DEPLOYMENT.md` - Documentation
- `scripts/` - Utility scripts
- Training logs and confusion matrices

---

## Troubleshooting

### Models Not Loading
- Check checkpoint paths match: `checkpoints/{model_name}/{model_name}_best.pt`
- Ensure model_metadata loads from: `results/{model_name}_metrics.json`

### Memory Issues
- Transformer is optimized for CPU (220K params)
- If using GPU Space, models will automatically use GPU
- All models fit in <2GB RAM

### Import Errors
- Ensure `requirements.txt` includes all dependencies
- Run `pip install -r requirements.txt` locally first to test

---

## requirements.txt for HF Spaces

```
torch==2.0.0
torchvision==0.15.0
numpy==1.24.0
pillow==9.5.0
scikit-learn==1.3.0
matplotlib==3.7.0
gradio==3.50.0
```

---

## Performance Notes

- **Cold Start Time**: ~10-15 seconds (model loading)
- **Inference Speed**: <100ms per model on Transformer
- **Memory**: ~1.5GB for all models loaded
- **Recommended Space**: free tier (CPU) or GPU if available

---

## Monitoring & Maintenance

Check Space runtime:
```bash
hf download AhmedSamir1598/AutoVision-Perception --repo-type=space
```

Update models:
```bash
git pull origin main
# Update checkpoints/
git add checkpoints/
git commit -m "Update trained models"
git push
```

---

## Next Steps

After deployment:
1. Share Space link with classmates/advisors
2. Add Space badge to main README
3. Monitor Space activity in HF Hub dashboard
4. Document any issues or improvements
