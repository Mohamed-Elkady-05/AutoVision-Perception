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
import math
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


class AdditiveAttention(nn.Module):
    """Lightweight additive attention for sequence pooling."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        # sequence_outputs: (batch, seq_len, hidden)
        weights = torch.softmax(self.score(sequence_outputs), dim=1)
        context = torch.sum(weights * sequence_outputs, dim=1)
        return context, weights


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for temporal transformers."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# Placeholder classes for students to implement
# These will be implemented by Team Members 2 and 3

class LSTMModel(BaseSequentialModel):
    """
    LSTM (Long Short-Term Memory) model for sequence classification.
    
    To be implemented by Student 2.
    Should include attention mechanism for better interpretability.
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = AdditiveAttention(lstm_output_size)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        hidden = F.relu(self.fc1(context))
        hidden = self.dropout_layer(hidden)
        logits = self.fc2(hidden)
        return logits
    
    def get_model_name(self) -> str:
        return "LSTM"
    
    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "model": "LSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "num_parameters": self.get_num_parameters(),
        }


class TransformerModel(BaseSequentialModel):
    """
    Transformer model for sequence classification.
    
    To be implemented by Student 3.
    Should include multi-head self-attention and optional Vision Transformer.
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
        attention_heads: int = 8,
        ffn_dim: int = 1024,
        max_seq_len: int = 512,
    ):
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout, device)

        self.bidirectional = bidirectional
        self.attention_heads = attention_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_size, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_seq_len={self.max_seq_len}")

        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout_layer(hidden)
        logits = self.fc2(hidden)
        return logits
    
    def get_model_name(self) -> str:
        return "Transformer"
    
    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "model": "Transformer",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "attention_heads": self.attention_heads,
            "ffn_dim": self.ffn_dim,
            "max_seq_len": self.max_seq_len,
            "num_parameters": self.get_num_parameters(),
        }


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
