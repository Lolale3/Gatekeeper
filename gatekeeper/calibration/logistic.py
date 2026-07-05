"""A pure-Python logistic-regression calibrator. It learns, from labeled data, a
mapping from a field's signal vector to a calibrated risk = P(this value is wrong).
No numpy / scikit-learn -- just the standard library -- so the whole package stays
dependency-free and the math is fully visible. The Calibrator interface is
swappable, so an sklearn / isotonic version can drop in for larger datasets.
"""
from __future__ import annotations

import math
from typing import Optional

from ..types import Record
from ..schema import Schema
from ..interfaces import Calibrator


def _sigmoid(z: float) -> float:
    # numerically stable logistic
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class LogisticCalibrator(Calibrator):
    """features = signal vector, target = 1 if the field was wrong else 0.
    Trained by batch gradient descent. Predicts P(error), stored as field.risk."""
    name = "logistic"

    def __init__(self, *, lr: float = 0.3, epochs: int = 3000, l2: float = 0.0,
                 default_signal: float = 0.5):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.default_signal = default_signal
        self.feature_names: list[str] = []
        self.weights: list[float] = []
        self.bias: float = 0.0
        self._fitted = False

    def _vectorize(self, signal_map: dict) -> list[float]:
        # consistent feature order; a missing signal falls back to a neutral value
        return [float(signal_map.get(name, self.default_signal)) for name in self.feature_names]

    def fit(self, X: list[dict], y: list) -> "LogisticCalibrator":
        names: set = set()
        for row in X:
            names.update(row.keys())
        self.feature_names = sorted(names)
        rows = [self._vectorize(row) for row in X]
        m = len(self.feature_names)
        n = len(rows)
        self.weights = [0.0] * m
        self.bias = 0.0
        if n == 0:
            self._fitted = True
            return self
        for _ in range(self.epochs):
            grad_w = [0.0] * m
            grad_b = 0.0
            for xi, yi in zip(rows, y):
                p = _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, xi)))
                err = p - float(yi)
                for j in range(m):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(m):
                self.weights[j] -= self.lr * (grad_w[j] / n + self.l2 * self.weights[j])
            self.bias -= self.lr * (grad_b / n)
        self._fitted = True
        return self

    def predict_proba(self, signal_map: dict) -> float:
        x = self._vectorize(signal_map)
        return _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, x)))

    def weights_map(self) -> dict:
        return dict(zip(self.feature_names, self.weights))

    def calibrate(self, record: Record, schema: Schema) -> Record:
        for field in record.fields.values():
            if self._fitted and self.feature_names:
                field.risk = self.predict_proba(field.signal_vector())
            else:
                vals = list(field.signal_vector().values())     # unfitted fallback
                field.risk = 1.0 - (sum(vals) / len(vals) if vals else 0.0)
        return record