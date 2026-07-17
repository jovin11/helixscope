from pathlib import Path

import numpy as np
from pssp_model import ProteinStructurePredictor


class FakeModel:
    def predict(self, inputs, **_kwargs):
        assert inputs.shape == (1, 8, 20)
        output = np.zeros((1, 8, 3), dtype=np.float32)
        output[0, :, 2] = 0.8
        output[0, 0] = (0.9, 0.05, 0.05)
        output[0, 1] = (0.05, 0.85, 0.1)
        return output


def test_predictor_lazy_loads_once_and_returns_summary(tmp_path: Path):
    model_path = tmp_path / "model.keras"
    model_path.touch()
    load_calls = []

    def loader(path):
        load_calls.append(path)
        return FakeModel()

    predictor = ProteinStructurePredictor(model_path, max_length=8, model_loader=loader)

    assert predictor.status()["loaded"] is False
    first = predictor.predict("ACD")
    second = predictor.predict("AC")

    assert len(load_calls) == 1
    assert first.structure == "HEC"
    assert first.to_dict()["summary"]["class_counts"] == {
        "helix": 1,
        "sheet": 1,
        "coil": 1,
    }
    assert second.structure == "HE"
    assert predictor.status()["loaded"] is True
