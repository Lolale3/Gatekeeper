"""Load the fixture invoices, split them deterministically, and demonstrate the
type-aware correctness function on a few right/wrong extractions.

    PYTHONPATH=. python examples/dataset_demo.py
"""
from gatekeeper import FixtureInvoiceLoader, invoice_schema, split_examples, is_correct


def main() -> None:
    schema = invoice_schema()
    examples = FixtureInvoiceLoader().load()
    print(f"Loaded {len(examples)} labeled invoices "
          f"(schema fields: {', '.join(schema.field_names())}).\n")

    splits = split_examples(examples)
    print("Deterministic splits:", splits.summary(), "\n")

    ex = examples[3]                      # inv-004, an absent-vendor case
    print(f"Example {ex.id}:")
    print("  source_text:", repr(ex.source_text))
    print("  ground_truth:", ex.ground_truth.values, "\n")

    print("Type-aware correctness (predicted vs gold):")
    checks = [
        ("total_amount",   "number", "$1,240.00",            "1240.00"),
        ("total_amount",   "number", "1240",                 "1250.00"),
        ("invoice_date",   "date",   "14/03/2026",           "2026-03-14"),
        ("invoice_date",   "date",   "2026-03-15",           "2026-03-14"),
        ("invoice_number", "exact",  "4471",                 "4471"),
        ("invoice_number", "exact",  "4471",                 "4472"),
        ("vendor_name",    "str",    "ACME FREIGHT CO.",     "Acme Freight Co."),
        ("vendor_name",    "str",    "Acme Freight Company", "Acme Freight Co."),
        ("vendor_name",    "str",    "Northwind",            None),
    ]
    for field, dtype, pred, gold in checks:
        ok = is_correct(pred, gold, dtype)
        mark = "OK" if ok else "XX"
        print(f"  [{mark}] {field:15}({dtype:6}) {str(pred)!r:24} vs {gold!r}")


if __name__ == "__main__":
    main()