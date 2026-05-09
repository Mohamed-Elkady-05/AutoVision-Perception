"""
GRU Model Training - GTSRB Traffic Sign Classification
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from config import TrainingConfig
from sequence_dataset import create_sequence_dataloaders
from base_sequential_models import GRUModel
from unified_trainer import UnifiedTrainer


FEATURES_DIR   = "/content/drive/MyDrive/gtsrb_cache/npy"
RESULTS_DIR    = "./results"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE     = 32
NUM_WORKERS    = 0
NUM_EPOCHS     = 30


# Wrapper that casts sequences to float32 without breaking len()
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
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRUModel(
        input_size    = 512,
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        output_size   = 43,
        dropout       = dropout,
        bidirectional = bidirectional,
        device        = device,
    )

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