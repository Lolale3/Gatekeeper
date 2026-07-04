"""Core data model for the gatekeeper confidence-gating pipeline.

A Record flows through the pipeline accumulating information at each stage: the
extractor fills in field values, signal generators annotate fields with
confidence signals, the calibrator adds a risk estimate, and the policy adds a
decision. Nothing is discarded -- the Record carries its full provenance, which
is what later gives us auditability (§6 feedback, §7 eval) for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Optional


class Action(str, Enum):
    APPROVE = "approve"
    ESCALATE = "escalate"


@dataclass
class Signal:
    """A single confidence signal attached to a field by one generator."""
    name: str                       # e.g. "self_consistency", "logprob", "rule_check"
    value: float                    # convention: higher = more confident, 0..1
    generator: str                  # which SignalGenerator produced it
    meta: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class Decision:
    """The approve/escalate outcome for a field or a whole record."""
    action: Action
    reason: str = ""
    threshold: Optional[float] = None
    risk: Optional[float] = None


@dataclass
class Field:
    """One extracted value plus everything the pipeline learns about it."""
    name: str
    value: Any = None
    source_span: Optional[str] = None       # text the value came from (§3 grounding)
    signals: list[Signal] = dc_field(default_factory=list)
    risk: Optional[float] = None            # calibrated P(value is wrong), 0..1
    decision: Optional[Decision] = None

    def add_signal(self, signal: Signal) -> None:
        self.signals.append(signal)

    def get_signal(self, name: str) -> Optional[Signal]:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def signal_vector(self) -> dict[str, float]:
        """Signals as a {name: value} map -- the calibrator's input features."""
        return {s.name: s.value for s in self.signals}


@dataclass
class Record:
    """One document (invoice, freight email) moving through the pipeline."""
    id: str
    source_text: str
    fields: dict[str, Field] = dc_field(default_factory=dict)
    decision: Optional[Decision] = None     # record-level roll-up
    meta: dict[str, Any] = dc_field(default_factory=dict)

    def ensure_field(self, name: str) -> Field:
        if name not in self.fields:
            self.fields[name] = Field(name=name)
        return self.fields[name]

    def set_value(self, name: str, value: Any, source_span: Optional[str] = None) -> Field:
        f = self.ensure_field(name)
        f.value = value
        f.source_span = source_span
        return f

    def get_field(self, name: str) -> Optional[Field]:
        return self.fields.get(name)