"""The orchestrator. Deliberately dumb: it holds the four components and passes
a Record through them in order. All intelligence lives in the components.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Record
from .schema import Schema
from .interfaces import Extractor, SignalGenerator, Calibrator, Policy


@dataclass
class Pipeline:
    schema: Schema
    extractor: Extractor
    signal_generators: list[SignalGenerator]
    calibrator: Calibrator
    policy: Policy

    def run(self, record: Record) -> Record:
        record = self.extractor.extract(record, self.schema)
        for gen in self.signal_generators:
            record = gen.generate(record, self.schema)
        record = self.calibrator.calibrate(record, self.schema)
        record = self.policy.decide(record, self.schema)
        return record

    def run_batch(self, records: list[Record]) -> list[Record]:
        return [self.run(r) for r in records]