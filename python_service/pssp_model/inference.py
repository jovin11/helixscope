"""Lazy model loading and the complete serving-time prediction flow."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol

import numpy as np

from .preprocessing import decode_probabilities, encode_sequence, normalize_sequence


class PredictiveModel(Protocol):
    def predict(self, inputs: np.ndarray, **kwargs: Any) -> np.ndarray: ...


ModelLoader = Callable[[Path], PredictiveModel]


@dataclass(frozen=True)
class Prediction:
    sequence: str
    structure: str
    confidence: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        counts = Counter(self.structure)
        length = len(self.sequence)
        return {
            "sequence": self.sequence,
            "prediction": self.structure,
            "confidence": [round(value, 4) for value in self.confidence],
            "summary": {
                "length": length,
                "mean_confidence": round(sum(self.confidence) / length, 4),
                "class_counts": {
                    "helix": counts["H"],
                    "sheet": counts["E"],
                    "coil": counts["C"],
                },
            },
        }


class ProteinStructurePredictor:
    """Validate, encode, and predict Q3 secondary structure for one sequence."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_length: int = 400,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_length = max_length
        self._model_loader = model_loader or self._load_keras_model
        self._model: PredictiveModel | None = None
        self._load_lock = Lock()

    @staticmethod
    def _load_keras_model(model_path: Path) -> PredictiveModel:
        from tensorflow.keras.models import load_model

        return load_model(model_path, compile=False)

    def _get_model(self) -> PredictiveModel:
        # Double-checked locking
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    if not self.model_path.is_file():
                        raise FileNotFoundError(self.model_path)
                    self._model = self._model_loader(self.model_path)
        return self._model

    def status(self) -> dict[str, Any]:
        return {
            "available": self.model_path.is_file(),
            "loaded": self._model is not None,
            "max_sequence_length": self.max_length,
            "classes": ["helix", "sheet", "coil"],
        }

    def predict(self, raw_sequence: str) -> Prediction:
        sequence = normalize_sequence(raw_sequence, max_length=self.max_length)
        batch = np.expand_dims(
            encode_sequence(sequence, max_length=self.max_length), axis=0
        )
        model = self._get_model()
        try:
            raw_prediction = model.predict(batch, verbose=0)
        except TypeError:
            raw_prediction = model.predict(batch)

        prediction = np.asarray(raw_prediction)
        if prediction.ndim != 3 or prediction.shape[0] != 1:
            raise ValueError(
                "Expected model output shape (1, sequence_length, 3), "
                f"received {prediction.shape}."
            )
        structure, confidence = decode_probabilities(
            prediction[0], sequence_length=len(sequence)
        )
        return Prediction(sequence, structure, confidence)
