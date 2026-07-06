"""The LLMExtractor: model-agnostic logic that turns source_text into a populated
Record. It builds a prompt from the schema, calls a client in JSON mode, parses
defensively, and populates the Record with values and best-effort source spans.
Correctness of the *values* is deliberately NOT this layer's job -- that is the
confidence layer (§3-§5). This layer only makes the FORMAT trustworthy.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..types import Record
from ..schema import Schema
from ..interfaces import Extractor
from .client import LLMClient

_DTYPE_HINT = {
    "str": "text",
    "exact": "the exact identifier or code, copied as written",
    "number": "a numeric value only, no currency symbols or thousands separators (expand shorthand like '42k' to 42000)",
    "date": "a date in YYYY-MM-DD format",
}


def build_prompt(schema: Schema, source_text: str) -> str:
    lines = [
        "You are a precise data-extraction system.",
        "Extract the following fields from the document and return ONLY a JSON "
        "object with exactly these keys:",
    ]
    for spec in schema.fields:
        lines.append(f'  "{spec.name}": {_DTYPE_HINT.get(spec.dtype, "text")}')
    lines += [
        "",
        "If a field is not present in the document, set its value to null.",
        "Return only the JSON object -- no explanation, no markdown.",
        "",
        "DOCUMENT:",
        source_text,
    ]
    return "\n".join(lines)


def parse_json_object(raw: str) -> dict:
    """Recover a JSON object from a raw model response: handles clean JSON,
    markdown-fenced JSON, and JSON wrapped in prose. Raises ValueError if none
    can be recovered (the extractor turns that into graceful degradation)."""
    if not raw or not raw.strip():
        raise ValueError("empty response")
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError(f"could not parse a JSON object from response: {raw[:80]!r}")


def locate_span(source_text: str, value: Any) -> Optional[str]:
    """Best-effort provenance: find the extracted value in the source and return
    the matched substring, or None if it isn't there (a hallucination hint for
    §3). This is a locate, not a correctness judgement."""
    if value is None:
        return None
    needle = str(value).strip()
    if not needle:
        return None
    idx = source_text.lower().find(needle.lower())
    if idx == -1:
        return None
    return source_text[idx:idx + len(needle)]


def _norm_key(k: str) -> str:
    return re.sub(r"[\s_]+", "", str(k).strip().lower())


def _clean_value(v: Any) -> Any:
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return v


class LLMExtractor(Extractor):
    def __init__(self, client: LLMClient, *, temperature: float = 0.0):
        self.client = client
        self.temperature = temperature

    def extract(self, record: Record, schema: Schema) -> Record:
        prompt = build_prompt(schema, record.source_text)
        try:
            raw = self.client.generate(prompt, temperature=self.temperature, json_mode=True)
            parsed = parse_json_object(raw)
        except Exception as e:
            record.meta["extraction_error"] = str(e)    # never crash the pipeline
            parsed = {}
        self._populate(record, schema, parsed)
        return record

    def _populate(self, record: Record, schema: Schema, parsed: dict) -> None:
        by_key = {_norm_key(k): val for k, val in parsed.items()}
        for spec in schema.fields:
            value = _clean_value(by_key.get(_norm_key(spec.name)))
            span = locate_span(record.source_text, value)
            record.set_value(spec.name, value=value, source_span=span)