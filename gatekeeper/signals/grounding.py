"""Grounding signal: is the extracted value actually supported by the source text,
judged AFTER type-aware normalization? This is the fix for the literal-find false
alarm from §2 -- a correct normalized number like 1240 for '$1,240.00', or a date
'2026-03-03' for 'March 3, 2026', now reads as grounded instead of 'not-found'.
A value that cannot be found in the source at all scores 0 (a hallucination hint).
"""
from __future__ import annotations

import re

from ..types import Record, Signal
from ..schema import Schema
from ..interfaces import SignalGenerator
from ..data.matching import to_number, to_date, _norm_str

_NUM_TOKEN = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_DATE_TOKEN = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"
)


def _source_numbers(source: str) -> set:
    out = set()
    for tok in _NUM_TOKEN.findall(source):
        v = to_number(tok)
        if v is not None:
            out.add(v)
    return out


def _source_dates(source: str) -> set:
    out = set()
    for tok in _DATE_TOKEN.findall(source):
        d = to_date(tok)
        if d is not None:
            out.add(d)
    return out


class GroundingSignal(SignalGenerator):
    name = "grounding"

    def generate(self, record: Record, schema: Schema) -> Record:
        norm_source = _norm_str(record.source_text)
        src_numbers = _source_numbers(record.source_text)
        src_dates = _source_dates(record.source_text)
        for spec in schema.fields:
            value = record.fields[spec.name].value
            score = self._ground(value, spec.dtype, norm_source, src_numbers, src_dates)
            record.fields[spec.name].add_signal(
                Signal(name=self.name, value=score, generator=self.name)
            )
        return record

    @staticmethod
    def _ground(value, dtype, norm_source, src_numbers, src_dates) -> float:
        if value is None:
            return 1.0                       # nothing claimed -> nothing invented
        if dtype == "number":
            v = to_number(value)
            return 1.0 if v is not None and any(abs(v - s) < 1e-6 for s in src_numbers) else 0.0
        if dtype == "date":
            d = to_date(value)
            return 1.0 if d is not None and d in src_dates else 0.0
        nv = _norm_str(value)                # str / exact
        return 1.0 if nv and nv in norm_source else 0.0