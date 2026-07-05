"""The feedback loop. Items the system escalated get corrected by a human; those
corrections are fresh ground-truth labels. Instead of discarding them, we fold
them into the calibration data and refit -- so the calibrator sharpens over time.

Escalated items are, by construction, the ones the system was least sure about,
so they carry the most information per label (active learning). Corrections flow
into the CALIBRATION data only; the test split stays frozen, or the before/after
measurement would be measuring improvement on data it trained on.
"""
from __future__ import annotations

from ..types import Record, Action
from ..schema import Schema
from ..data.matching import is_correct
from ..calibration import LogisticCalibrator


def corrections_to_training_data(records, corrections, schema: Schema):
    """Turn (escalated records, human corrections) into (X, y) calibration
    instances. Only fields the correction actually labels are used; a field whose
    extracted value differs from the human's correct value is labeled an error."""
    X, y = [], []
    for record, corr in zip(records, corrections):
        for spec in schema.fields:
            if not corr.has(spec.name):          # only fields the human labeled
                continue
            field = record.fields[spec.name]
            X.append(field.signal_vector())
            y.append(0 if is_correct(field.value, corr.get(spec.name), spec.dtype) else 1)
    return X, y


def select_escalated_records(records):
    """The records the policy routed to a human (record-level ESCALATE)."""
    return [r for r in records if r.decision is not None and r.decision.action is Action.ESCALATE]


class FeedbackLoop:
    """Accumulates human-corrected labels on top of a base calibration set and
    refits the calibrator on the combined data."""

    def __init__(self, base_X, base_y, *, calibrator_factory=LogisticCalibrator):
        self.base_X = list(base_X)
        self.base_y = list(base_y)
        self._fb_X: list = []
        self._fb_y: list = []
        self.calibrator_factory = calibrator_factory

    def add_corrections(self, records, corrections, schema: Schema) -> int:
        Xn, yn = corrections_to_training_data(records, corrections, schema)
        self._fb_X.extend(Xn)
        self._fb_y.extend(yn)
        return len(Xn)

    def add_labeled(self, X, y) -> int:
        self._fb_X.extend(X)
        self._fb_y.extend(y)
        return len(list(X))

    def training_data(self):
        return self.base_X + self._fb_X, self.base_y + self._fb_y

    def fit(self):
        X, y = self.training_data()
        return self.calibrator_factory().fit(X, y)

    @property
    def feedback_count(self) -> int:
        return len(self._fb_y)