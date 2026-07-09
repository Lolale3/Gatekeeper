"""Two-temperature stability signal (fragility). Sample the extractor at a LOW and a
HIGH temperature and check, per field, whether the consensus answer survives the
temperature change. A field whose value shifts when sampling heats up is fragile ->
lower confidence.

This is a cousin of self-consistency: self-consistency resamples one temperature and
compares to the primary answer; this contrasts two temperatures. Because both are
sampling-based value-agreement signals (we have no token log-probs), they may be
correlated -- the eval ablation is what honestly settles whether this adds signal on
top. Following practitioner guidance, the 'high' temperature is moderate, not extreme
(too high makes even correct answers look unstable).
"""
from __future__ import annotations

from ..types import Record, Signal
from ..schema import Schema
from ..interfaces import SignalGenerator, Extractor
from ..data.matching import is_correct


def _majority(values, dtype):
    """The value the most others agree with, under type-aware equality."""
    best, best_count = None, -1
    for v in values:
        c = sum(1 for u in values if is_correct(u, v, dtype))
        if c > best_count:
            best, best_count = v, c
    return best


class TwoTemperatureSignal(SignalGenerator):
    name = "two_temperature"

    def __init__(self, low_extractor: Extractor, high_extractor: Extractor, *, n: int = 1):
        # low_extractor / high_extractor should be built at low and (moderate) high temps.
        self.low = low_extractor
        self.high = high_extractor
        self.n = n

    def _sample(self, extractor: Extractor, record: Record, schema: Schema):
        out = []
        for _ in range(self.n):
            fresh = Record(id=record.id, source_text=record.source_text)
            out.append(extractor.extract(fresh, schema))
        return out

    def generate(self, record: Record, schema: Schema) -> Record:
        low = self._sample(self.low, record, schema)
        high = self._sample(self.high, record, schema)
        for spec in schema.fields:
            v_low = _majority([s.fields[spec.name].value for s in low], spec.dtype)
            v_high = _majority([s.fields[spec.name].value for s in high], spec.dtype)
            stable = is_correct(v_low, v_high, spec.dtype)
            record.fields[spec.name].add_signal(
                Signal(name=self.name, value=1.0 if stable else 0.0, generator=self.name,
                       meta={"low": v_low, "high": v_high})
            )
        return record