"""Turning labeled records into calibrator training data, plus the calibration
measurement -- reliability table and Expected Calibration Error (ECE) -- used to
verify that the calibrated risk is actually honest (predicted vs actual error rate).
"""
from __future__ import annotations

from ..schema import Schema
from ..data.matching import is_correct


def build_training_data(records, ground_truths, schema: Schema):
    """From records that already carry signals + their gold answers, produce
    (X, y): X = list of per-field signal vectors, y = 1 if the field was wrong."""
    X, y = [], []
    for record, gt in zip(records, ground_truths):
        for spec in schema.fields:
            field = record.fields[spec.name]
            X.append(field.signal_vector())
            correct = is_correct(field.value, gt.get(spec.name), spec.dtype)
            y.append(0 if correct else 1)
    return X, y


def naive_risk(signal_map: dict) -> float:
    """Baseline 'confidence' with no calibration: 1 - mean(signals)."""
    vals = list(signal_map.values())
    return 1.0 - (sum(vals) / len(vals) if vals else 0.0)


def reliability_table(predictions, labels, n_bins: int = 5):
    """Bin predictions and compare predicted risk vs actual error rate per bin."""
    bins = [[] for _ in range(n_bins)]
    for p, actual in zip(predictions, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, actual))
    rows = []
    for i, b in enumerate(bins):
        if b:
            rows.append({
                "lo": i / n_bins, "hi": (i + 1) / n_bins, "count": len(b),
                "mean_pred": sum(p for p, _ in b) / len(b),
                "actual": sum(a for _, a in b) / len(b),
            })
        else:
            rows.append({"lo": i / n_bins, "hi": (i + 1) / n_bins, "count": 0,
                         "mean_pred": None, "actual": None})
    return rows


def expected_calibration_error(predictions, labels, n_bins: int = 5) -> float:
    n = len(predictions)
    if n == 0:
        return 0.0
    ece = 0.0
    for row in reliability_table(predictions, labels, n_bins):
        if row["count"]:
            ece += (row["count"] / n) * abs(row["mean_pred"] - row["actual"])
    return ece