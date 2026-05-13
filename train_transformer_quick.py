"""Quick Transformer training test."""
import torch
from pathlib import Path
from src.config import TrainingConfig
from src.detection.sequence_dataset import create_sequence_dataloaders
from src.models.base_sequential_models import TransformerModel
from src.models.unified_trainer import UnifiedTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load data
train_loader, val_loader, test_loader = create_sequence_dataloaders(
    features_dir="./cache/vgg16_sequence_features",
    batch_size=32,
    num_workers=0,
    augment=True,
    seed=42,
)

# Build optimized Transformer
model = TransformerModel(input_size=512, output_size=43, device=device)
print(f"Model: {model.get_model_name()}")
print(f"Parameters: {model.get_num_parameters():,}")

# Train
config = TrainingConfig()
config.NUM_EPOCHS = 15  # Fewer epochs for quick test
config.EARLY_STOPPING_PATIENCE = 5
config.DEVICE = device

trainer = UnifiedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device,
    save_dir="checkpoints/transformer",
)

print("\n[Training Transformer...]")
trainer.train(num_epochs=15)

# Evaluate
metrics = trainer.evaluate(test_loader)
trainer.save_training_curves("results")
trainer.save_metrics_json(metrics, "results")

print("\n[Final Metrics]")
for key, value in metrics.items():
    if key != "confusion_matrix":
        print(f"  {key}: {value:.4f}")

print("\nTransformer training completed successfully!")
