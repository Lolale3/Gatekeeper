"""What a labeled example looks like.

A LabeledExample keeps two things deliberately separate:
  - source_text : the raw document the extractor will read (the input)
  - ground_truth: the answer key, field_name -> correct value

The Record from the pipeline holds what the *system produced*; GroundTruth holds
what is *correct*. The eval harness (§7) and calibrator (§4) line the two up to
decide, per field, whether the extraction was right.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

from ..types import Record


@dataclass
class GroundTruth:
    """The answer key for one document: field_name -> correct value.

    A value of None means the field is genuinely ABSENT from the document -- a
    real and important case, because an extractor that invents a value for a
    field that isn't there has made an error we want to catch."""
    values: dict[str, Any] = dc_field(default_factory=dict)

    def get(self, field_name: str) -> Any:
        return self.values.get(field_name)

    def has(self, field_name: str) -> bool:
        return field_name in self.values


@dataclass
class LabeledExample:
    """One document paired with its answer key."""
    id: str
    source_text: str
    ground_truth: GroundTruth
    meta: dict[str, Any] = dc_field(default_factory=dict)

    def to_record(self) -> Record:
        """A fresh Record carrying only the input -- no fields, no answers. The
        extractor (§2) fills the fields in; ground_truth is never copied in, so
        the pipeline can never 'cheat' by seeing the gold values."""
        return Record(id=self.id, source_text=self.source_text, meta=dict(self.meta))