"""Hand-built labeled invoices -- the fixture dataset for §1.

These let us build and test the whole data layer offline, with zero download
friction, exactly as the stubs did for §0. They are deliberately varied: several
date formats, totals with and without currency symbols and commas, near-duplicate
identifiers, and two documents with an ABSENT vendor (gold = None) to exercise
the hallucination/omission cases. A real DocILE-style adapter replaces these as
the source of real error distributions once the machinery is proven.
"""
from __future__ import annotations

from ..schema import Schema, FieldSpec
from .example import GroundTruth, LabeledExample
from .loader import DatasetLoader
from .matching import to_number, to_date


# --- business rules attached to the invoice schema (used by the §3 RuleSignal) ---
# Each rule takes (value, fields) and returns True if the value passes. None
# (an absent field) always passes -- absence is not a malformed value.

def _no_leading_punctuation(value, fields=None) -> bool:
    """An identifier shouldn't start with stray punctuation -- catches '#4471'."""
    if value is None:
        return True
    s = str(value).strip()
    return bool(s) and s[0].isalnum()


def _positive_amount(value, fields=None) -> bool:
    if value is None:
        return True
    v = to_number(value)
    return v is not None and v > 0


def _plausible_year(value, fields=None) -> bool:
    if value is None:
        return True
    d = to_date(value)
    return d is None or (2000 <= d.year <= 2035)


def invoice_schema() -> Schema:
    """Canonical invoice schema used by the fixtures and the demo."""
    return Schema(
        doc_type="invoice",
        fields=[
            FieldSpec("invoice_number", dtype="exact",  critical=True,  error_cost=3.0,
                      rules=[_no_leading_punctuation]),
            FieldSpec("invoice_date",   dtype="date",   critical=False, error_cost=1.0,
                      rules=[_plausible_year]),
            FieldSpec("total_amount",   dtype="number", critical=True,  error_cost=5.0,
                      rules=[_positive_amount]),
            FieldSpec("vendor_name",    dtype="str",    critical=False, error_cost=1.0),
        ],
    )


_FIXTURES = [
    ("inv-001",
     "INVOICE #4471\nDate: 2026-03-14\nTotal Due: $1,240.00\nFrom: Acme Freight Co.",
     {"invoice_number": "4471", "invoice_date": "2026-03-14",
      "total_amount": "1240.00", "vendor_name": "Acme Freight Co."}),
    ("inv-002",
     "Invoice No: INV-2025-0088\nInvoice Date: 05/11/2025\nAmount Payable: 2,500.00\n"
     "Vendor: Blue Ridge Logistics LLC",
     {"invoice_number": "INV-2025-0088", "invoice_date": "2025-11-05",
      "total_amount": "2500.00", "vendor_name": "Blue Ridge Logistics LLC"}),
    ("inv-003",
     "RECEIPT\nRef 7789-A\nDated: March 3, 2026\nGrand Total USD 875.50\n"
     "Merchant: Cardinal Supply",
     {"invoice_number": "7789-A", "invoice_date": "2026-03-03",
      "total_amount": "875.50", "vendor_name": "Cardinal Supply"}),
    ("inv-004",
     "Invoice 100234\n2026/01/09\nTotal: 12,000\n(no vendor listed)",
     {"invoice_number": "100234", "invoice_date": "2026-01-09",
      "total_amount": "12000", "vendor_name": None}),
    ("inv-005",
     "FACTURE / INVOICE\nNumber: 55-C\n1 Feb 2026\nTotal amount: $ 640.00\n"
     "Supplier: Northwind Carriers",
     {"invoice_number": "55-C", "invoice_date": "2026-02-01",
      "total_amount": "640.00", "vendor_name": "Northwind Carriers"}),
    ("inv-006",
     "Invoice#: 9001\nDate 03.11.2026\nBalance: 1,999.99\nFrom Evergreen Transport Inc.",
     {"invoice_number": "9001", "invoice_date": "2026-11-03",
      "total_amount": "1999.99", "vendor_name": "Evergreen Transport Inc."}),
    ("inv-007",
     "Statement\nDoc No: A-4471\nInvoice Date: 2026-06-30\nPay: $ 3,150.00\n"
     "Billed by: Summit Freight Solutions",
     {"invoice_number": "A-4471", "invoice_date": "2026-06-30",
      "total_amount": "3150.00", "vendor_name": "Summit Freight Solutions"}),
    ("inv-008",
     "INVOICE\nNo 6620\nDate: 07/04/2026\nTotal Due: 480\nVendor: Harbor Point Shipping",
     {"invoice_number": "6620", "invoice_date": "2026-04-07",
      "total_amount": "480", "vendor_name": "Harbor Point Shipping"}),
    ("inv-009",
     "Tax Invoice 33219\nDated 22 Dec 2025\nAmount: $9,875.25\nParkway Logistics Group",
     {"invoice_number": "33219", "invoice_date": "2025-12-22",
      "total_amount": "9875.25", "vendor_name": "Parkway Logistics Group"}),
    ("inv-010",
     "Invoice number: X-7\n2026-02-28\nTotal: 55.00\n(receipt, walk-in)",
     {"invoice_number": "X-7", "invoice_date": "2026-02-28",
      "total_amount": "55.00", "vendor_name": None}),
    ("inv-011",
     "INVOICE #4472\nInvoice Date: April 5, 2026\nTotal Payable: $1,240.00\n"
     "From: Acme Freight Co.",
     {"invoice_number": "4472", "invoice_date": "2026-04-05",
      "total_amount": "1240.00", "vendor_name": "Acme Freight Co."}),
    ("inv-012",
     "Billing Summary\nRef: 8080\nDate: 2026-05-19\nGrand total: USD 2,340.75\n"
     "Carrier: Ironclad Freightways",
     {"invoice_number": "8080", "invoice_date": "2026-05-19",
      "total_amount": "2340.75", "vendor_name": "Ironclad Freightways"}),
]


class FixtureInvoiceLoader(DatasetLoader):
    name = "fixture_invoices"

    def load(self) -> list[LabeledExample]:
        return [
            LabeledExample(
                id=ex_id,
                source_text=text,
                ground_truth=GroundTruth(values=dict(gold)),
            )
            for ex_id, text, gold in _FIXTURES
        ]