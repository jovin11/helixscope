"""Model architecture, preprocessing, training, and inference for HelixScope."""

from .inference import Prediction, ProteinStructurePredictor
from .preprocessing import (
    AMINO_ACIDS,
    STRUCTURE_CLASSES,
    SequenceValidationError,
    decode_probabilities,
    decode_structures,
    encode_sequence,
    encode_structures,
    normalize_sequence,
    preprocess_sequences,
    preprocess_structures,
)

__all__ = [
    "AMINO_ACIDS",
    "STRUCTURE_CLASSES",
    "Prediction",
    "ProteinStructurePredictor",
    "SequenceValidationError",
    "decode_probabilities",
    "decode_structures",
    "encode_sequence",
    "encode_structures",
    "normalize_sequence",
    "preprocess_sequences",
    "preprocess_structures",
]
