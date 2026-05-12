"""Transformer training smoke test for GTSRB sequence loading."""

from __future__ import annotations

from base_sequential_models import TransformerModel
from config import SequentialModelConfig, TransformerConfig
from sequence_model_smoke import run_sequence_model_smoke_test


def main() -> None:
    seq_config = SequentialModelConfig()
    transformer_config = TransformerConfig()
    run_sequence_model_smoke_test(
        model_name="Transformer",
        model_factory=lambda device: TransformerModel(
            input_size=seq_config.INPUT_SIZE,
            hidden_size=seq_config.HIDDEN_SIZE,
            num_layers=transformer_config.NUM_TRANSFORMER_LAYERS,
            output_size=seq_config.OUTPUT_SIZE,
            dropout=transformer_config.DROPOUT,
            bidirectional=seq_config.BIDIRECTIONAL,
            device=device,
            attention_heads=transformer_config.ATTENTION_HEADS,
            ffn_dim=transformer_config.FFN_DIM,
            max_seq_len=256,
        ),
        epochs=50,
    )


if __name__ == "__main__":
    main()
