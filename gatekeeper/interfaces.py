"""The four contracts. The pipeline knows only these interfaces, never the
concrete implementations -- that is what makes every stage swappable (local
model vs Gemini extractor, one signal vs five, stub calibrator vs a fitted one).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .types import Record
from .schema import Schema


class Extractor(ABC):
    """Turn a raw Record (source_text only) into one with fields populated."""

    @abstractmethod
    def extract(self, record: Record, schema: Schema) -> Record: ...


class SignalGenerator(ABC):
    """Annotate a Record's fields with one or more confidence signals."""

    name: str = "signal_generator"

    @abstractmethod
    def generate(self, record: Record, schema: Schema) -> Record: ...


class Calibrator(ABC):
    """Map a field's raw signals to a calibrated risk (P(value is wrong))."""

    def fit(self, records: Iterable[Record], labels) -> "Calibrator":
        # Optional: identity/stub calibrators need no fitting.
        return self

    @abstractmethod
    def calibrate(self, record: Record, schema: Schema) -> Record: ...


class Policy(ABC):
    """Turn calibrated risk (+ cost params) into approve/escalate decisions."""

    @abstractmethod
    def decide(self, record: Record, schema: Schema) -> Record: ...