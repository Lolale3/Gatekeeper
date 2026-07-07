"""Real-invoice adapter: loads mychen76/invoices-and-receipts_ocr_v1 from Hugging
Face into LabeledExample objects. The OCR word list becomes source_text (messy,
real OCR ordering); the annotated header/summary become the ground truth, mapped
onto the invoice schema. Unlike the fixtures and the freight generator, neither the
OCR noise nor the model's errors on it are authored by us -- this is what makes the
resulting risk-coverage numbers honest.

Needs the datasets library:  pip install datasets   (or  pip install -e ".[real]")
The import is lazy, so the rest of the package works without it.
"""
from __future__ import annotations

import ast
import json
import re

from .example import GroundTruth, LabeledExample
from .loader import DatasetLoader

HF_NAME = "mychen76/invoices-and-receipts_ocr_v1"


def _company_name(seller):
    """The dataset's 'seller' is company name + full address concatenated. Grading
    vendor_name against the whole thing is unfair (the model extracts the name), so
    we cut at the first standalone number -- where the street address usually starts."""
    if not seller:
        return seller
    s = str(seller).strip()
    m = re.search(r"\s\d", s)
    return s[:m.start()].strip() if m else s


def _to_obj(x):
    """Coerce a possibly stringified structure into a Python object. The dataset
    stores nested data as strings (JSON, and Python-repr with single quotes), so we
    try both parsers."""
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return x
    s = x.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(s)
        except Exception:
            continue
    return s


def parse_row(raw_data, parsed_data):
    """One dataset row -> (source_text, gold_dict) mapped to the invoice schema."""
    raw = _to_obj(raw_data)
    ocr = raw.get("ocr_words") if isinstance(raw, dict) else raw
    ocr = _to_obj(ocr)
    source_text = "\n".join(str(w) for w in ocr) if isinstance(ocr, list) else str(ocr)

    labels = _to_obj(parsed_data)
    if isinstance(labels, dict) and "json" in labels:
        labels = _to_obj(labels["json"])
    header = labels.get("header", {}) if isinstance(labels, dict) else {}
    summary = labels.get("summary", {}) if isinstance(labels, dict) else {}

    gold = {
        "invoice_number": header.get("invoice_no"),
        "invoice_date": header.get("invoice_date"),
        "vendor_name": _company_name(header.get("seller")),
        "total_amount": (summary.get("total_gross_worth")
                         or summary.get("total")
                         or summary.get("gross_worth")),
    }
    return source_text, gold


class RealInvoiceLoader(DatasetLoader):
    name = "real_invoices"

    def __init__(self, split: str = "train", limit: int | None = 120, hf_name: str = HF_NAME):
        self.split = split
        self.limit = limit
        self.hf_name = hf_name

    def load(self):
        from datasets import load_dataset          # lazy: only needed if actually used
        ds = load_dataset(self.hf_name, split=self.split)
        out = []
        for i, row in enumerate(ds):
            source_text, gold = parse_row(row.get("raw_data"), row.get("parsed_data"))
            # keep rows that have the anchor fields; skip badly-parsed ones
            if not source_text.strip() or not gold.get("invoice_number"):
                continue
            out.append(LabeledExample(
                id=str(row.get("id", f"inv-{i:04d}")),
                source_text=source_text,
                ground_truth=GroundTruth(values=gold),
            ))
            if self.limit and len(out) >= self.limit:
                break
        return out