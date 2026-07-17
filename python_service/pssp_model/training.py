"""Explicit training and fine-tuning entry points for prepared NumPy datasets.

Dataset collection is deliberately outside CI. Callers provide arrays shaped
``(samples, sequence_length, 20)`` and ``(samples, sequence_length, 3)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import build_model


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: str | Path = "best_model_v2.keras",
    validation_fraction: float = 0.2,
    epochs: int = 50,
    batch_size: int = 64,
):
    """Build and train a model using the historical training recipe."""

    if features.ndim != 3 or labels.ndim != 3:
        raise ValueError("features and labels must both be rank-3 arrays")
    if len(features) != len(labels) or len(features) < 2:
        raise ValueError("features and labels need the same sample count (at least 2)")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    split_index = int((1 - validation_fraction) * len(features))
    if split_index == 0 or split_index == len(features):
        raise ValueError("validation_fraction produces an empty dataset split")

    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    model = build_model(
        sequence_length=features.shape[1],
        num_features=features.shape[2],
        num_classes=labels.shape[2],
    )
    callbacks = [
        ModelCheckpoint(
            str(output_path), monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        features[:split_index],
        labels[:split_index],
        validation_data=(features[split_index:], labels[split_index:]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    return model, history


def fine_tune_model(
    model_path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
):
    """Continue training an existing checkpoint on prepared labeled examples."""

    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    history = model.fit(features, labels, epochs=epochs, batch_size=batch_size)
    model.save(output_path)
    return model, history
