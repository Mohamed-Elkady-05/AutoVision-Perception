"""GRU training smoke test for GTSRB sequence loading."""

from __future__ import annotations

from base_sequential_models import GRUModel
from config import SequentialModelConfig
from sequence_model_smoke import run_sequence_model_smoke_test


def main() -> None:
    config = SequentialModelConfig()
    run_sequence_model_smoke_test(
        model_name="GRU",
        model_factory=lambda device: GRUModel(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            output_size=config.OUTPUT_SIZE,
            dropout=config.DROPOUT,
            bidirectional=config.BIDIRECTIONAL,
            device=device,
        ),
        epochs=50,
    )


if __name__ == "__main__":
    main()
