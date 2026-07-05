"""Business-rule signal: run type/format validity plus the schema's custom field
rules. Cheap (no model calls) and often the strongest, most controllable signal
-- e.g. the leading-punctuation rule that catches the stray '#' on '#4471' that
grounding was blind to. The score is the fraction of checks that passed.
"""
from __future__ import annotations

from typing import Any

from ..types import Record, Signal
from ..schema import Schema
from ..interfaces import SignalGenerator
from ..data.matching import to_number, to_date


def dtype_valid(value: Any, dtype: str) -> bool:
    """Is the value structurally valid for its declared type?"""
    if value is None:
        return True                          # absence is not a malformed value
    if dtype == "number":
        return to_number(value) is not None
    if dtype == "date":
        return to_date(value) is not None
    return True                              # str / exact: always structurally valid


class RuleSignal(SignalGenerator):
    name = "rules"

    def generate(self, record: Record, schema: Schema) -> Record:
        for spec in schema.fields:
            field = record.fields[spec.name]
            checks = [dtype_valid(field.value, spec.dtype)]
            for rule in spec.rules:
                try:
                    checks.append(bool(rule(field.value, record.fields)))
                except Exception:
                    checks.append(False)     # a rule that errors counts as a failure
            score = sum(checks) / len(checks) if checks else 1.0
            field.add_signal(Signal(name=self.name, value=score, generator=self.name,
                                    meta={"passed": sum(checks), "total": len(checks)}))
        return record