from app import create_app
from pssp_model import Prediction, normalize_sequence


class StubPredictor:
    max_length = 12

    def status(self):
        return {
            "available": True,
            "loaded": True,
            "max_sequence_length": self.max_length,
            "classes": ["helix", "sheet", "coil"],
        }

    def predict(self, value):
        sequence = normalize_sequence(value, max_length=self.max_length)
        return Prediction(
            sequence=sequence,
            structure="H" * len(sequence),
            confidence=(0.9,) * len(sequence),
        )


def make_client():
    app = create_app({"TESTING": True}, predictor=StubPredictor())
    return app.test_client()


def test_health_endpoint_does_not_require_real_model():
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.get_json()["model"]["loaded"] is True


def test_prediction_endpoint_returns_structured_result():
    response = make_client().post("/api/v1/predict", json={"sequence": "ACD"})

    assert response.status_code == 200
    assert response.get_json() == {
        "sequence": "ACD",
        "prediction": "HHH",
        "confidence": [0.9, 0.9, 0.9],
        "summary": {
            "length": 3,
            "mean_confidence": 0.9,
            "class_counts": {"helix": 3, "sheet": 0, "coil": 0},
        },
    }


def test_prediction_endpoint_keeps_legacy_payload_compatible():
    response = make_client().post("/predict", json={"input_data": "AC"})

    assert response.status_code == 200
    assert response.get_json()["prediction"] == "HH"


def test_prediction_endpoint_rejects_invalid_sequence():
    response = make_client().post("/api/v1/predict", json={"sequence": "ACX"})

    assert response.status_code == 422
    assert response.get_json()["details"]["invalid_residues"] == ["X"]


def test_prediction_endpoint_rejects_non_json_body():
    response = make_client().post("/api/v1/predict", data="ACD")

    assert response.status_code == 400
