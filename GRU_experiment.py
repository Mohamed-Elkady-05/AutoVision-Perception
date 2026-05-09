"""
GRU Model Training - GTSRB Traffic Sign Classification

Usage in notebook:
    from gru_training import load_data, build_model, train_model, evaluate_model

    train_loader, val_loader, test_loader = load_data()
    model                                 = build_model()
    trainer                               = train_model(model, train_loader, val_loader)
    metrics                               = evaluate_model(trainer, test_loader)
"""

import torch
from pathlib import Path

from config import TrainingConfig
from sequence_dataset import create_sequence_dataloaders
from base_sequential_models import GRUModel
from unified_trainer import UnifiedTrainer


# ==============================================================================
# DEFAULT SETTINGS
# Can be overridden by passing arguments to each function
# ==============================================================================

FEATURES_DIR   = "/content/drive/MyDrive/gtsrb_cache"
RESULTS_DIR    = "./results"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE     = 32
NUM_WORKERS    = 0
NUM_EPOCHS     = 30


# ==============================================================================
# FUNCTIONS
# ==============================================================================

def load_data(
    features_dir = FEATURES_DIR,
    batch_size   = BATCH_SIZE,
    num_workers  = NUM_WORKERS,
):
    """
    Load the dataset and return the three DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    
    Example:
        train_loader, val_loader, test_loader = load_data()
    """
    train_loader, val_loader, test_loader = create_sequence_dataloaders(
        features_dir = features_dir,
        batch_size   = batch_size,
        num_workers  = num_workers,
        augment      = True,
        seed         = 42,
    )

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
    """
    Build and return the GRU model.

    Args:
        hidden_size   : GRU internal memory size (try 128, 256, 512)
        num_layers    : number of stacked GRU layers (try 1, 2, 3)
        bidirectional : read sequence forwards and backwards
        dropout       : dropout probability

    Returns:
        model (GRUModel)

    Example:
        model = build_model()
        model = build_model(hidden_size=512, num_layers=3)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRUModel(
        input_size    = 512,        # VGG16 output size, fixed
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        output_size   = 43,         # 43 GTSRB classes, fixed
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
    """
    Train the GRU model and return the trainer (which holds the full history).

    Args:
        model         : GRUModel returned by build_model()
        train_loader  : training DataLoader from load_data()
        val_loader    : validation DataLoader from load_data()
        num_epochs    : max number of training epochs
        learning_rate : optimizer learning rate
        weight_decay  : L2 regularization strength
        checkpoint_dir: where to save the best model weights

    Returns:
        trainer (UnifiedTrainer)
            trainer.history        -> loss and accuracy per epoch
            trainer.best_val_acc   -> best validation accuracy achieved
            trainer.best_epoch     -> epoch at which best val acc occurred

    Example:
        trainer = train_model(model, train_loader, val_loader)
        print(trainer.best_val_acc)
        print(trainer.history["val_acc"])
    """
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
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        config         = config,
        device         = device,
        save_dir       = checkpoint_dir,
    )

    trainer.train(num_epochs=num_epochs)

    return trainer


def evaluate_model(
    trainer,
    test_loader,
    results_dir = RESULTS_DIR,
):
    """
    Evaluate the trained model on the test set and save results.

    Args:
        trainer     : UnifiedTrainer returned by train_model()
        test_loader : test DataLoader from load_data()
        results_dir : where to save the plots and metrics JSON

    Returns:
        metrics (dict)
            metrics["accuracy"]         -> overall accuracy
            metrics["precision"]        -> macro precision
            metrics["recall"]           -> macro recall
            metrics["f1"]               -> macro F1 score
            metrics["confusion_matrix"] -> confusion matrix array

    Example:
        metrics = evaluate_model(trainer, test_loader)
        print(metrics["accuracy"])
        print(metrics["confusion_matrix"])
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    metrics = trainer.evaluate(test_loader)
    trainer.save_training_curves(results_dir)
    trainer.save_metrics_json(metrics, results_dir)

    print(f"\nResults saved to: {results_dir}/")

    return metrics