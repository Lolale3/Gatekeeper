"""gatekeeper.calibration -- §4: turn a field's signal vector into a calibrated
risk = P(value is wrong), learned from labeled data, and measure whether it's honest."""
from .logistic import LogisticCalibrator
from .metrics import (
    build_training_data, naive_risk, reliability_table, expected_calibration_error,
)

__all__ = [
    "LogisticCalibrator",
    "build_training_data", "naive_risk", "reliability_table",
    "expected_calibration_error",
]