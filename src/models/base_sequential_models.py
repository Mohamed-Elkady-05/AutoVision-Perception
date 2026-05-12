"""
Base Sequential Model Architecture
Common interface and utilities for RNN, GRU, LSTM, and Transformer models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseSequentialModel(ABC, nn.Module):
    """Abstract base class for sequential classifiers."""

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_size: int = 43,
        dropout: float = 0.3,
        device: str = "cuda",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.device = device

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_config_dict(self) -> Dict[str, Any]:
        pass

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_trainable_parameters(self) -> int:
        return self.get_num_parameters()


class RNNModel(BaseSequentialModel):
    """Vanilla RNN model for sequence classification."""

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
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(out_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        rnn_out, _ = self.rnn(x)
        pooled = rnn_out.mean(dim=1)
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout_layer(hidden)
        return self.fc2(hidden)

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
    """GRU model for sequence classification."""

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
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(out_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        gru_out, _ = self.gru(x)
        pooled = gru_out.mean(dim=1)
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout_layer(hidden)
        return self.fc2(hidden)

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
    """Additive attention for sequence pooling."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        weights = torch.softmax(self.score(sequence_outputs), dim=1)
        context = torch.sum(weights * sequence_outputs, dim=1)
        return context, weights


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class LSTMModel(BaseSequentialModel):
    """LSTM model with additive attention."""

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
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.attention = AdditiveAttention(out_size)
        self.fc1 = nn.Linear(out_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        hidden = F.relu(self.fc1(context))
        hidden = self.dropout_layer(hidden)
        return self.fc2(hidden)

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
    """Transformer encoder model for sequence classification."""

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
        return self.fc2(hidden)

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


def create_model(model_name: str, config: Dict[str, Any], device: str = "cuda") -> BaseSequentialModel:
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


def get_model_summary(model: BaseSequentialModel) -> Dict[str, Any]:
    return {
        "name": model.get_model_name(),
        "config": model.get_config_dict(),
        "total_params": model.get_num_parameters(),
        "trainable_params": model.get_trainable_parameters(),
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 10, 512).to(device)

    rnn_model = RNNModel(input_size=512, hidden_size=256, num_layers=2, output_size=43, device=device).to(device)
    rnn_out = rnn_model(x)
    print(f"RNN output shape: {rnn_out.shape}")

    gru_model = GRUModel(input_size=512, hidden_size=256, num_layers=2, output_size=43, device=device).to(device)
    gru_out = gru_model(x)
    print(f"GRU output shape: {gru_out.shape}")

    print("\n✓ Base sequential models tests passed!")


# Backward-compatible aliases for older imports in notebooks/scripts.
RNN_Model = RNNModel
GRU_Model = GRUModel
