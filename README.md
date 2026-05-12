---
title: AutoVision Perception
author: Copilot
emoji: "🚦"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# AutoVision Perception

GTSRB traffic sign inference with a Gradio front end and a backend that can load image and temporal sequence models.

## Overview

This repo treats GTSRB as a sequence problem by turning ordered images from each class folder into fixed-length pseudo-sequences. The images are not true video frames; they are grouped by class, sorted within each folder, and sliced into overlapping windows so the sequential models receive consistent, class-specific frame blocks.

The pipeline is split into three stages:

1. Build pseudo-sequences from the extracted archive.
2. Precompute VGG16 features for every frame in each sequence.
3. Train the RNN on the cached sequence features and visualize predictions on the saved frame strips.

## How the sequential data is built

The preprocessing scripts read images from `archive/Train/<class>/`, sort the files inside each class folder, and take sliding windows of length `DatasetConfig.SEQUENCE_LENGTH` with stride `DatasetConfig.FRAME_STRIDE`.

Each saved `.npz` file contains:

- `frames`: the stacked images for one pseudo-sequence.
- `metadata`: JSON metadata with the class label, frame indices, timestamps, and the original folder name.

The next stage computes VGG16 features for each frame in the saved sequence while preserving the sequence structure. Those feature files are what `sequence_dataset.py` loads during training.

## Local setup

```bash
pip install -r requirements.txt
```

If you are using the Conda environment from the project, you can also run the scripts with:

```bash
conda run -n projects python <script>.py
```

## Generate the sequence data

Create pseudo-sequences from the extracted GTSRB archive:

```bash
python preprocess_gtsrb_sequences_fast.py
```

That script writes sequence `.npz` files under `data/preprocessed_sequences/`.

Then precompute the VGG16 feature cache used by the RNN:

```bash
python preprocess_gtsrb_sequences.py
```

You can also run the preprocessing and feature extraction in one pass with the full script above, since it calls the feature precomputer after building the sequences.

## Environment variables

Set these in the Space settings when you add weights:

- `GTSRB_CNN_CHECKPOINT`
- `GTSRB_SEQUENCE_CHECKPOINT`
- `GTSRB_SEQUENCE_MODEL` (`rnn`, `gru`, `lstm`, or `transformer` when available)
- `GTSRB_HF_REPO_ID`
- `GTSRB_CNN_FILENAME`
- `GTSRB_SEQUENCE_FILENAME`
- `HF_TOKEN`

## Train the RNN

```bash
python rnn_training_smoke.py
```

That training entrypoint uses the cached real sequences in `cache/vgg16_sequence_features/`, trains the RNN for 50 epochs, and writes the best checkpoint plus evaluation plots to `checkpoints/` and `results/`.

## Run inference and visualize results

For a quick inference pass over the saved real sequences and a visual check of the outputs:

```bash
python visualize_rnn_sequences.py
```

That script loads the trained RNN checkpoint, runs predictions on the saved frame strips, and writes a montage to `results/RNN_sequence_visualizations.png`.

If you want the Gradio demo, launch:

```bash
python app.py
```

## Notes

- The repository uses GTSRB as the core dataset.
- Temporal inputs should always be related frames from the same source clip or the same ordered class folder, not random windows.
- Sequence preprocessing and dataset loading are separated from the UI so they can be swapped for production checkpoints later.
- Generated artifacts such as `data/`, `cache/`, `results/`, `checkpoints/`, and `archive.zip` are ignored by Git.
