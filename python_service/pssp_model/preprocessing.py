"""Validation and fixed-width residue encoding used for training and inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STRUCTURE_CLASSES = ("H", "E", "C")
_AMINO_ACID_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
_STRUCTURE_INDEX = {label: index for index, label in enumerate(STRUCTURE_CLASSES)}


class SequenceValidationError(ValueError):
    """Raised when an amino-acid sequence cannot be used for prediction."""

    def __init__(
        self,
        message: str,
        *,
        invalid_residues: Sequence[str] = (),
        max_length: int | None = None,
    ) -> None:
        super().__init__(message)
        self.invalid_residues = tuple(invalid_residues)
        self.max_length = max_length

    def to_dict(self) -> dict[str, Any]:
        details: dict[str, Any] = {}
        if self.invalid_residues:
            details["invalid_residues"] = list(self.invalid_residues)
        if self.max_length is not None:
            details["max_length"] = self.max_length
        return details


def normalize_sequence(raw_sequence: str, *, max_length: int = 400) -> str:
    """Normalize plain text or one FASTA record and validate canonical residues."""

    if not isinstance(raw_sequence, str):
        raise SequenceValidationError("Sequence must be provided as text.")

    lines = [line.strip() for line in raw_sequence.strip().splitlines()]
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    sequence = "".join("".join(lines).split()).upper()

    if not sequence:
        raise SequenceValidationError("Enter at least one amino-acid residue.")
    if len(sequence) > max_length:
        raise SequenceValidationError(
            f"Sequence is {len(sequence)} residues; the model supports at most "
            f"{max_length}.",
            max_length=max_length,
        )

    invalid_residues = sorted(set(sequence) - set(AMINO_ACIDS))
    if invalid_residues:
        symbols = ", ".join(invalid_residues)
        raise SequenceValidationError(
            f"Unsupported residue symbol(s): {symbols}.",
            invalid_residues=invalid_residues,
        )
    return sequence


def encode_sequence(sequence: str, *, max_length: int = 400) -> NDArray[np.float32]:
    """One-hot encode a validated sequence and zero-pad it to the model width."""

    if len(sequence) > max_length:
        raise ValueError(f"Sequence length exceeds max_length={max_length}.")
    encoding = np.zeros((max_length, len(AMINO_ACIDS)), dtype=np.float32)
    for position, residue in enumerate(sequence):
        try:
            encoding[position, _AMINO_ACID_INDEX[residue]] = 1.0
        except KeyError as exc:
            raise ValueError(f"Unsupported amino acid: {residue}") from exc
    return encoding


def encode_structures(
    structure: str, *, max_length: int = 400
) -> NDArray[np.float32]:
    """One-hot encode Q3 labels for model training."""

    if len(structure) > max_length:
        raise ValueError(f"Structure length exceeds max_length={max_length}.")
    encoding = np.zeros((max_length, len(STRUCTURE_CLASSES)), dtype=np.float32)
    for position, label in enumerate(structure):
        try:
            encoding[position, _STRUCTURE_INDEX[label]] = 1.0
        except KeyError as exc:
            raise ValueError(f"Unsupported structure label: {label}") from exc
    return encoding


def decode_probabilities(
    probabilities: NDArray[np.floating], *, sequence_length: int
) -> tuple[str, tuple[float, ...]]:
    """Decode per-residue probabilities into Q3 labels and confidences."""

    matrix = np.asarray(probabilities)
    if matrix.ndim != 2 or matrix.shape[1] != len(STRUCTURE_CLASSES):
        raise ValueError(
            f"Expected prediction shape (sequence_length, 3), received {matrix.shape}."
        )
    if matrix.shape[0] < sequence_length:
        raise ValueError("Model returned fewer positions than the input sequence.")

    trimmed = matrix[:sequence_length]
    indices = np.argmax(trimmed, axis=-1)
    structure = "".join(STRUCTURE_CLASSES[index] for index in indices)
    confidence = tuple(float(value) for value in np.max(trimmed, axis=-1))
    return structure, confidence


def preprocess_sequences(sequence: str, max_length: int = 400):
    """Compatibility alias for the original Flask integration."""

    return encode_sequence(sequence, max_length=max_length)


def preprocess_structures(structure: str, max_length: int = 400):
    """Compatibility alias for the original training script."""

    return encode_structures(structure, max_length=max_length)


def decode_structures(encoding) -> str:
    """Compatibility decoder returning only Q3 labels."""

    matrix = np.asarray(encoding)
    return "".join(STRUCTURE_CLASSES[index] for index in np.argmax(matrix, axis=-1))
