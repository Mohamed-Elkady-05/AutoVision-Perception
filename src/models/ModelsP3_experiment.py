"""
GRU Model Training - GTSRB Traffic Sign Classification
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.config import TrainingConfig
from src.detection.sequence_dataset import create_sequence_dataloaders
from src.models.base_sequential_models import GRUModel, RNNModel, LSTMModel, TransformerModel
from src.models.unified_trainer import UnifiedTrainer


FEATURES_DIR   = "./cache/vgg16_sequence_features"
RESULTS_DIR    = "./results"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE     = 32
NUM_WORKERS    = 0
NUM_EPOCHS     = 30


class Float32Loader:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for sequences, labels in self.loader:
            yield sequences.float(), labels


def load_data(
    features_dir = FEATURES_DIR,
    batch_size   = BATCH_SIZE,
    num_workers  = NUM_WORKERS,
):
    train_loader, val_loader, test_loader = create_sequence_dataloaders(
        features_dir = features_dir,
        batch_size   = batch_size,
        num_workers  = num_workers,
        augment      = True,
        seed         = 42,
    )

    train_loader = Float32Loader(train_loader)
    val_loader   = Float32Loader(val_loader)
    test_loader  = Float32Loader(test_loader)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    return train_loader, val_loader, test_loader


def build_model(
    hidden_size   = 256,
    num_layers    = 2,
    bidirectional = True,
    dropout       = 0.3,
    model_type     = "gru",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type.lower() == "gru":
        model = GRUModel(
            input_size    = 512,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            output_size   = 43,
            dropout       = dropout,
            bidirectional = bidirectional,
            device        = device,
        )
    elif model_type.lower() == "rnn":
        model = RNNModel(
            input_size    = 512,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            output_size   = 43,
            dropout       = dropout,
            bidirectional = bidirectional,
            device        = device,
        )
    elif model_type.lower() == "lstm":
        model = LSTMModel(
            input_size    = 512,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            output_size   = 43,
            dropout       = dropout,
            bidirectional = bidirectional,
            device        = device,
        )
    elif model_type.lower() == "transformer":
        model = TransformerModel(
            input_size    = 512,
            output_size   = 43,
            device        = device,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Model      : {model.get_model_name()}")
    print(f"Parameters : {model.get_num_parameters():,}")
    print(f"Device     : {device}")

    return model


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs     = NUM_EPOCHS,
    learning_rate  = 1e-3,
    weight_decay   = 1e-4,
    checkpoint_dir = CHECKPOINT_DIR,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    config = TrainingConfig()
    config.LEARNING_RATE            = learning_rate
    config.WEIGHT_DECAY             = weight_decay
    config.NUM_EPOCHS               = num_epochs
    config.EARLY_STOPPING_PATIENCE  = 10
    config.SCHEDULER_TYPE           = "cosine"
    config.SAVE_BEST_ONLY           = True
    config.DEVICE                   = device

    trainer = UnifiedTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = config,
        device       = device,
        save_dir     = checkpoint_dir,
    )

    trainer.train(num_epochs=num_epochs)

    return trainer


def evaluate_model(
    trainer,
    test_loader,
    results_dir = RESULTS_DIR,
):
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    metrics = trainer.evaluate(test_loader)
    trainer.save_training_curves(results_dir)
    trainer.save_metrics_json(metrics, results_dir)

    print(f"\nResults saved to: {results_dir}/")

    return metrics


def main():
    """Train all 4 models (RNN, GRU, LSTM, Transformer) sequentially."""
    import json
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("ModelsP3 FINAL TRAINING EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Load data once (reuse across all models)
    print("\n[Loading data...]")
    train_loader, val_loader, test_loader = load_data(
        features_dir=FEATURES_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    
    model_types = ["rnn", "gru", "lstm", "transformer"]
    results_summary = {}
    
    for model_type in model_types:
        print("\n" + "=" * 80)
        print(f"TRAINING {model_type.upper()}")
        print("=" * 80)
        
        # Build model
        model = build_model(model_type=model_type)
        
        # Train model
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            checkpoint_dir=f"{CHECKPOINT_DIR}/{model_type}",
        )
        
        # Evaluate model
        metrics = evaluate_model(
            trainer=trainer,
            test_loader=test_loader,
            results_dir=RESULTS_DIR,
        )
        
        # Save confusion matrix PNG
        if trainer.last_eval_targets and trainer.last_eval_preds:
            cm = confusion_matrix(trainer.last_eval_targets, trainer.last_eval_preds)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, cmap="Blues")
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{model_type.upper()} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            output_path = Path(RESULTS_DIR) / f"{model_type.upper()}_confusion_matrix.png"
            fig.tight_layout()
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved confusion matrix: {output_path}")
        
        # Store results for summary
        results_summary[model_type.upper()] = {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "loss": metrics.get("loss"),
        }
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    for model_name, model_metrics in results_summary.items():
        print(f"\n{model_name}:")
        for metric_name, metric_value in model_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save summary to JSON
    summary_path = Path(RESULTS_DIR) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
