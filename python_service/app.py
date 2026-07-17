"""Flask application for protein secondary-structure inference.

The application uses an app factory so it can be tested without loading the
TensorFlow model. The production predictor loads its model lazily on the first
prediction request, keeping health checks fast and inexpensive.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from pssp_model import ProteinStructurePredictor
from pssp_model import SequenceValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "best_model_v2.keras"


def create_app(
    config: dict[str, Any] | None = None,
    *,
    predictor: ProteinStructurePredictor | None = None,
) -> Flask:
    """Create and configure the web application.

    A predictor may be injected by tests or alternate deployments. When it is
    omitted, the configured Keras model is loaded only when inference begins.
    """

    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "public" / "templates"),
        static_folder=str(PROJECT_ROOT / "public" / "static"),
    )
    app.config.from_mapping(
        MODEL_PATH=os.getenv("PSSP_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        MAX_SEQUENCE_LENGTH=int(os.getenv("PSSP_MAX_SEQUENCE_LENGTH", "400")),
        MAX_CONTENT_LENGTH=64 * 1024,
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

    prediction_service = predictor or ProteinStructurePredictor(
        model_path=app.config["MODEL_PATH"],
        max_length=app.config["MAX_SEQUENCE_LENGTH"],
    )
    app.extensions["prediction_service"] = prediction_service

    @app.get("/")
    def home() -> str:
        return render_template(
            "pssp.html", max_sequence_length=prediction_service.max_length
        )

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "helixscope",
                "model": prediction_service.status(),
            }
        )

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"error": "Request body exceeds the 64 KiB limit."}), 413

    def predict_response():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        # ``input_data`` remains supported for clients of the original API.
        sequence = payload.get("sequence", payload.get("input_data"))
        if not isinstance(sequence, str):
            return jsonify({"error": "The 'sequence' field must be a string."}), 400

        try:
            result = prediction_service.predict(sequence)
        except SequenceValidationError as exc:
            return jsonify({"error": str(exc), "details": exc.to_dict()}), 422
        except FileNotFoundError:
            app.logger.exception("Configured model file was not found")
            return jsonify({"error": "The prediction model is unavailable."}), 503
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify({"error": "Prediction failed. Please try again."}), 500

        return jsonify(result.to_dict())

    app.add_url_rule(
        "/api/v1/predict", "predict_v1", predict_response, methods=["POST"]
    )
    app.add_url_rule("/predict", "predict_legacy", predict_response, methods=["POST"])

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("PSSP_HOST", "127.0.0.1"),
        port=int(os.getenv("PSSP_PORT", "5001")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
