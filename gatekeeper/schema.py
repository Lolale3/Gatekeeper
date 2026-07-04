"""Schema: which fields we expect to extract, and the constraints on them.

The Schema does double duty -- it tells the extractor what to look for, and it
is the source of the business-rule signals used in §3. A FieldSpec also carries
the criticality and economic weight (error_cost) that the cost-aware policy in
§5 will use to decide what is worth escalating.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Callable, Optional

# A rule takes a field's value (and optionally the record's field map) and
# returns True if it passes. Real rules arrive in §3; the Schema just holds them.
Rule = Callable[..., bool]


@dataclass
class FieldSpec:
    name: str
    dtype: str = "str"                 # "str" | "number" | "date" | ...
    required: bool = True
    critical: bool = False             # low confidence here escalates the whole record
    error_cost: float = 1.0            # relative $ cost of getting this field wrong (§5)
    rules: list[Rule] = dc_field(default_factory=list)


@dataclass
class Schema:
    doc_type: str
    fields: list[FieldSpec] = dc_field(default_factory=list)

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def get_spec(self, name: str) -> Optional[FieldSpec]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def critical_fields(self) -> list[str]:
        return [f.name for f in self.fields if f.critical]