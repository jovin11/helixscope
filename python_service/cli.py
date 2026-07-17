"""Command-line inference entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pssp_model import ProteinStructurePredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict Q3 protein secondary structure with the CNN-LSTM model."
    )
    parser.add_argument("sequence", help="Amino-acid sequence or single-record FASTA")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            os.getenv("PSSP_MODEL_PATH", PROJECT_ROOT / "best_model_v2.keras")
        ),
        help="Path to a trained .keras model",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = ProteinStructurePredictor(args.model).predict(args.sequence)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(f"Sequence  {result.sequence}")
    print(f"Structure {result.structure}")


if __name__ == "__main__":
    main()
