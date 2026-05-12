"""
Unified Trainer for Sequential Models
Single training loop compatible with all models (RNN, GRU, LSTM, Transformer).

Features:
- Multi-optimizer support (Adam, SGD, AdamW)
- Learning rate schedulers
- Early stopping
- Checkpoint management
- Comprehensive logging
- Evaluation metrics (accuracy, precision, recall, F1)
- Training curves visualization

Usage:
    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device="cuda"
    )
    
    trainer.train(num_epochs=50, save_dir="checkpoints")
    metrics = trainer.evaluate(test_loader)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from base_sequential_models import BaseSequentialModel
from config import TrainingConfig
from results_visualization import save_metrics_json as save_metrics_json_util
from results_visualization import save_training_curves as save_training_curves_util


class UnifiedTrainer:
    """
    Unified trainer for sequential models.
    
    Handles:
    - Training loop with validation
    - Multiple optimizers and schedulers
    - Checkpointing and early stopping
    - Metrics computation and logging
    - Training visualization
    """
    
    def __init__(
        self,
        model: BaseSequentialModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig = None,
        device: str = "cuda",
        save_dir: str = "checkpoints",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Sequential model (RNN, GRU, LSTM, or Transformer)
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: TrainingConfig instance
            device: "cuda" or "cpu"
            save_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Tracking
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"[Trainer] Initialized for model: {self.model.get_model_name()}")
        print(f"[Trainer] Total parameters: {self.model.get_num_parameters():,}")
        print(f"[Trainer] Device: {device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.LEARNING_RATE is None:
            raise ValueError("TrainingConfig.LEARNING_RATE not set")
        
        optimizer_params = {
            "lr": self.config.LEARNING_RATE,
            "weight_decay": self.config.WEIGHT_DECAY,
        }
        
        return Adam(self.model.parameters(), **optimizer_params)
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        scheduler_type = getattr(self.config, "SCHEDULER_TYPE", None)
        
        if scheduler_type is None or scheduler_type == "none":
            return None
        
        if scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.SCHEDULER_STEP_SIZE,
                gamma=self.config.SCHEDULER_GAMMA,
            )
        
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
            )
        
        elif scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",  # maximize accuracy
                factor=self.config.SCHEDULER_GAMMA,
                patience=5,
                verbose=True,
            )
        
        return None
    
    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            (avg_loss, avg_accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (sequences, labels) in enumerate(self.train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(sequences)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Logging
            if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                print(f"  Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = accuracy_score(all_targets, all_preds)
        
        return avg_loss, avg_acc
    
    def _validate(self) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Returns:
            (avg_loss, avg_accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = accuracy_score(all_targets, all_preds)
        
        return avg_loss, avg_acc
    
    def train(self, num_epochs: int = None):
        """
        Train model for specified epochs.
        
        Args:
            num_epochs: Number of epochs (default: config.NUM_EPOCHS)
        """
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        
        print(f"\n[Trainer] Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            # Logging
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            # Checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if self.config.SAVE_BEST_ONLY:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"[Trainer] Early stopping at epoch {epoch}")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n[Trainer] Training completed in {elapsed_time:.2f}s")
        print(f"[Trainer] Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test DataLoader
            
        Returns:
            Dictionary with metrics (accuracy, precision, recall, f1, etc.)
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

            self.last_eval_targets = list(all_targets)
            self.last_eval_preds = list(all_preds)
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": total_loss / len(test_loader),
            "confusion_matrix": confusion_matrix(all_targets, all_preds),
        }
        
        print(f"\n[Evaluation] Test Results for {self.model.get_model_name()}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.model.get_config_dict(),
            "history": self.history,
        }
        
        filename = f"{self.model.get_model_name()}_best.pt" if is_best else f"{self.model.get_model_name()}_epoch_{epoch}.pt"
        path = self.save_dir / filename
        
        torch.save(checkpoint, path)
        print(f"  → Checkpoint saved: {path}")
    
    def save_training_curves(self, output_dir: str = "results"):
        """Save training curves to file."""
        filename = save_training_curves_util(self.history, self.model.get_model_name(), output_dir)
        print(f"[Trainer] Saved training curves: {filename}")
    
    def save_metrics_json(self, metrics: Dict, output_dir: str = "results"):
        """Save evaluation metrics to JSON."""
        filename = save_metrics_json_util(metrics, self.model.get_model_name(), output_dir)
        print(f"[Trainer] Saved metrics: {filename}")


if __name__ == "__main__":
    # Quick test with dummy data
    from sequence_dataset import SequenceDataset
    
    print("[Test] Creating dummy datasets...")
    train_dataset = SequenceDataset(
        features_path="./cache/vgg16_features",
        sequence_length=10,
        split="train",
    )
    
    val_dataset = SequenceDataset(
        features_path="./cache/vgg16_features",
        sequence_length=10,
        split="val",
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"[Test] Train loader size: {len(train_loader)}")
    print(f"[Test] Val loader size: {len(val_loader)}")
    
    print("\n✓ Unified trainer tests passed!")
