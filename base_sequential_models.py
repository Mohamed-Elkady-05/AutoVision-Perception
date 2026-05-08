"""
Base Sequential Model Architecture
Common interface and utilities for RNN, GRU, LSTM, and Transformer models.

Design Philosophy:
- Unified interface: all models inherit from BaseSequentialModel
- Each student implements specific model (RNN, GRU, LSTM, Transformer)
- Common evaluation metrics, logging, checkpointing
- Easy model swapping in trainer
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseSequentialModel(ABC, nn.Module):
    """
    Abstract base class for all sequential models.
    
    Subclasses must implement:
    - forward(x): Main forward pass
    - get_model_name(): Return model identifier
    - get_config_dict(): Return configuration dict for logging
    """
    
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_size: int = 43,
        dropout: float = 0.3,
        device: str = "cuda",
    ):
        """
        Initialize base sequential model.
        
        Args:
            input_size: Input feature dimension (512 for VGG16)
            hidden_size: Hidden state dimension
            num_layers: Number of layers
            output_size: Number of output classes
            dropout: Dropout probability
            device: "cuda" or "cpu"
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.device = device
        
        # Will be set by subclass
        self.encoder = None
        self.classifier = None
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output logits of shape (batch_size, output_size)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return unique model identifier (e.g., 'RNN', 'LSTM', 'Transformer')."""
        pass
    
    @abstractmethod
    def get_config_dict(self) -> Dict[str, Any]:
        """Return configuration dict for logging/checkpointing."""
        pass
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_trainable_parameters(self) -> int:
        """Same as get_num_parameters for compatibility."""
        return self.get_num_parameters()


class RNNModel(BaseSequentialModel):
    """
    Vanilla RNN model for sequence classification.
    
    Architecture:
    - Input: (batch, seq_len, 512)
    - RNN layers (bidirectional optional)
    - Temporal pooling (mean/max over sequence)
    - FC classifier
    - Output: (batch, 43)
    """
    
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_size: int = 43,
        dropout: float = 0.3,
        bidirectional: bool = True,
        device: str = "cuda",
    ):
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.bidirectional = bidirectional
        
        # RNN encoder
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Pooling and classifier
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(rnn_output_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len, input_size)
            
        Returns:
            logits: (batch_size, output_size)
        """
        # RNN forward
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch, seq_len, hidden*2 if bi else hidden)
        
        # Mean pooling over sequence
        pooled = rnn_out.mean(dim=1)  # (batch, hidden*2)
        
        # Classifier
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout_layer(hidden)
        logits = self.fc2(hidden)
        
        return logits
    
    def get_model_name(self) -> str:
        return "RNN"
    
    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "model": "RNN",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "num_parameters": self.get_num_parameters(),
        }


class GRUModel(BaseSequentialModel):
    """
    GRU (Gated Recurrent Unit) model for sequence classification.
    
    Similar to RNN but with gating mechanism for better gradient flow.
    """
    
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_size: int = 43,
        dropout: float = 0.3,
        bidirectional: bool = True,
        device: str = "cuda",
    ):
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)
        
        self.bidirectional = bidirectional
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Pooling and classifier
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(gru_output_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len, input_size)
            
        Returns:
            logits: (batch_size, output_size)
        """
        # GRU forward
        gru_out, _ = self.gru(x)
        
        # Mean pooling
        pooled = gru_out.mean(dim=1)
        
        # Classifier
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout_layer(hidden)
        logits = self.fc2(hidden)
        
        return logits
    
    def get_model_name(self) -> str:
        return "GRU"
    
    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "model": "GRU",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "num_parameters": self.get_num_parameters(),
        }


# Placeholder classes for students to implement
# These will be implemented by Team Members 2 and 3

class LSTMModel(BaseSequentialModel):
    """
    LSTM (Long Short-Term Memory) model for sequence classification.
    
    To be implemented by Student 2.
    Should include attention mechanism for better interpretability.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("LSTMModel to be implemented by Student 2")
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("LSTMModel to be implemented by Student 2")
    
    def get_model_name(self) -> str:
        raise NotImplementedError("LSTMModel to be implemented by Student 2")
    
    def get_config_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("LSTMModel to be implemented by Student 2")


class TransformerModel(BaseSequentialModel):
    """
    Transformer model for sequence classification.
    
    To be implemented by Student 3.
    Should include multi-head self-attention and optional Vision Transformer.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("TransformerModel to be implemented by Student 3")
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TransformerModel to be implemented by Student 3")
    
    def get_model_name(self) -> str:
        raise NotImplementedError("TransformerModel to be implemented by Student 3")
    
    def get_config_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("TransformerModel to be implemented by Student 3")


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_name: str, config: Dict[str, Any], device: str = "cuda") -> BaseSequentialModel:
    """
    Factory function to create model instances.
    
    Args:
        model_name: One of ["rnn", "gru", "lstm", "transformer"]
        config: Configuration dict with model parameters
        device: "cuda" or "cpu"
        
    Returns:
        Model instance
    """
    model_map = {
        "rnn": RNNModel,
        "gru": GRUModel,
        "lstm": LSTMModel,
        "transformer": TransformerModel,
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    
    model_class = model_map[model_name.lower()]
    return model_class(**config, device=device)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_summary(model: BaseSequentialModel) -> Dict[str, Any]:
    """Generate summary of model architecture and parameters."""
    return {
        "name": model.get_model_name(),
        "config": model.get_config_dict(),
        "total_params": model.get_num_parameters(),
        "trainable_params": model.get_trainable_parameters(),
    }


if __name__ == "__main__":
    # Test RNN and GRU models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dummy input
    batch_size, seq_len, input_size = 4, 10, 512
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    
    # Test RNN
    rnn_model = RNNModel(input_size=input_size, hidden_size=256, num_layers=2, output_size=43, device=device)
    rnn_model = rnn_model.to(device)
    rnn_out = rnn_model(x)
    print(f"RNN output shape: {rnn_out.shape}")
    print(f"RNN config: {rnn_model.get_config_dict()}")
    
    # Test GRU
    gru_model = GRUModel(input_size=input_size, hidden_size=256, num_layers=2, output_size=43, device=device)
    gru_model = gru_model.to(device)
    gru_out = gru_model(x)
    print(f"GRU output shape: {gru_out.shape}")
    print(f"GRU config: {gru_model.get_config_dict()}")
    
    print("\n✓ Base sequential models tests passed!")
