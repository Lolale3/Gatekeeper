"""Save / load a fitted calibrator as JSON, so the app can fit once and gate many
documents later without refitting. Just the learned parameters -- no pickling."""
from __future__ import annotations

import json

from ..calibration import LogisticCalibrator


def save_calibrator(calibrator: LogisticCalibrator, path: str) -> str:
    with open(path, "w") as fh:
        json.dump({
            "feature_names": calibrator.feature_names,
            "weights": calibrator.weights,
            "bias": calibrator.bias,
        }, fh, indent=2)
    return path


def load_calibrator(path: str) -> LogisticCalibrator:
    with open(path) as fh:
        data = json.load(fh)
    cal = LogisticCalibrator()
    cal.feature_names = list(data["feature_names"])
    cal.weights = list(data["weights"])
    cal.bias = float(data["bias"])
    cal._fitted = True
    return cal