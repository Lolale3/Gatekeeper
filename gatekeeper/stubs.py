"""Stub implementations for the walking skeleton (§0).

None of these are intelligent. Their only job is to prove the contracts fit
together and a Record can travel the whole pipeline. We replace them one at a
time in later sections: real extractor (§2), real signals (§3), fitted
calibrator (§4), cost-aware policy (§5).
"""
from __future__ import annotations

from .types import Record, Signal, Decision, Action
from .schema import Schema
from .interfaces import Extractor, SignalGenerator, Calibrator, Policy


class StubExtractor(Extractor):
    """Fills every schema field with a placeholder value, as if extracted."""

    def extract(self, record: Record, schema: Schema) -> Record:
        for spec in schema.fields:
            record.set_value(
                spec.name,
                value=f"<{spec.name}>",
                source_span=f"stub span for {spec.name}",
            )
        return record


class ConstantSignalGenerator(SignalGenerator):
    """Attaches one fixed 'confidence' signal to every field."""

    name = "constant"

    def __init__(self, value: float = 0.7):
        self.value = value

    def generate(self, record: Record, schema: Schema) -> Record:
        for f in record.fields.values():
            f.add_signal(Signal(name="constant", value=self.value, generator=self.name))
        return record


class ConstantCalibrator(Calibrator):
    """Risk = 1 - mean(signal values). No real calibration yet (§4)."""

    def calibrate(self, record: Record, schema: Schema) -> Record:
        for f in record.fields.values():
            vals = list(f.signal_vector().values())
            mean_conf = sum(vals) / len(vals) if vals else 0.0
            f.risk = 1.0 - mean_conf
        return record


class ThresholdPolicy(Policy):
    """Per-field decision by fixed risk threshold; the record escalates if any
    *critical* field escalates. This is the simplest real version of §5."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, record: Record, schema: Schema) -> Record:
        critical = set(schema.critical_fields())
        escalate_record = False
        for name, f in record.fields.items():
            risk = f.risk if f.risk is not None else 1.0
            action = Action.ESCALATE if risk > self.threshold else Action.APPROVE
            f.decision = Decision(
                action=action,
                reason=f"risk {risk:.2f} vs threshold {self.threshold:.2f}",
                threshold=self.threshold,
                risk=risk,
            )
            if action is Action.ESCALATE and name in critical:
                escalate_record = True
        record.decision = Decision(
            action=Action.ESCALATE if escalate_record else Action.APPROVE,
            reason="a critical field was escalated" if escalate_record
                   else "all critical fields approved",
        )
        return record