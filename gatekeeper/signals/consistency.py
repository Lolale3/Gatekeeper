"""Self-consistency signal: resample the extractor N times and measure, per field,
how often the samples agree with the primary extraction. Model wavering -> lower
confidence. Reuses the same LLMClient (via a sampling extractor) and §1's
normalization -- agreement is judged with is_correct, so format-only variation
('1240' vs '1240.00') does NOT count as disagreement. This is the costly signal
(N extra model calls); N is configurable.
"""
from __future__ import annotations

from ..types import Record, Signal
from ..schema import Schema
from ..interfaces import SignalGenerator, Extractor
from ..data.matching import is_correct


class SelfConsistencySignal(SignalGenerator):
    name = "self_consistency"

    def __init__(self, extractor: Extractor, *, n: int = 5):
        # `extractor` should be built with a sampling temperature (> 0).
        self.extractor = extractor
        self.n = n

    def generate(self, record: Record, schema: Schema) -> Record:
        samples = []
        for _ in range(self.n):
            fresh = Record(id=record.id, source_text=record.source_text)
            samples.append(self.extractor.extract(fresh, schema))

        for spec in schema.fields:
            primary = record.fields[spec.name].value
            agree = sum(
                1 for s in samples
                if is_correct(s.fields[spec.name].value, primary, spec.dtype)
            )
            score = agree / self.n if self.n else 0.5
            record.fields[spec.name].add_signal(
                Signal(name=self.name, value=score, generator=self.name,
                       meta={"n": self.n, "agree": agree})
            )
        return record