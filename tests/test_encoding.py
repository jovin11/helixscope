import numpy as np
import pytest
from pssp_model import (
    AMINO_ACIDS,
    SequenceValidationError,
    decode_probabilities,
    encode_sequence,
    normalize_sequence,
)


def test_normalize_sequence_accepts_fasta_and_whitespace():
    assert normalize_sequence(">protein example\nacd ef\nGH") == "ACDEFGH"


@pytest.mark.parametrize("value", ["", "   ", ">header\n"])
def test_normalize_sequence_rejects_empty_input(value):
    with pytest.raises(SequenceValidationError, match="at least one"):
        normalize_sequence(value)


def test_normalize_sequence_reports_invalid_residues():
    with pytest.raises(SequenceValidationError) as error:
        normalize_sequence("ACDXZ")

    assert error.value.invalid_residues == ("X", "Z")


def test_normalize_sequence_enforces_model_length():
    with pytest.raises(SequenceValidationError) as error:
        normalize_sequence("A" * 5, max_length=4)

    assert error.value.max_length == 4


def test_encode_sequence_has_expected_shape_and_padding():
    encoded = encode_sequence("AC", max_length=4)

    assert encoded.shape == (4, len(AMINO_ACIDS))
    assert encoded.dtype == np.float32
    assert encoded[0, 0] == 1
    assert encoded[1, 1] == 1
    assert encoded[2:].sum() == 0


def test_decode_probabilities_returns_labels_and_confidence():
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]],
        dtype=np.float32,
    )

    structure, confidence = decode_probabilities(probabilities, sequence_length=3)

    assert structure == "HEC"
    assert confidence == pytest.approx((0.8, 0.7, 0.7))
