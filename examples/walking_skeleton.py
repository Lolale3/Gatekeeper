"""Run a Record end to end through the walking-skeleton pipeline.

Nothing here is intelligent yet -- the point is to prove a Record can travel the
whole pipeline and come out with a value, signals, a risk, and a decision on
every field, plus a record-level roll-up.

    PYTHONPATH=. python examples/walking_skeleton.py
"""
from gatekeeper import Record, Schema, FieldSpec, build_pipeline


def invoice_schema() -> Schema:
    return Schema(
        doc_type="invoice",
        fields=[
            FieldSpec("invoice_number", dtype="str", critical=True, error_cost=3.0),
            FieldSpec("invoice_date", dtype="date", critical=False, error_cost=1.0),
            FieldSpec("total_amount", dtype="number", critical=True, error_cost=5.0),
            FieldSpec("vendor_name", dtype="str", critical=False, error_cost=1.0),
        ],
    )


def main() -> None:
    schema = invoice_schema()
    pipeline = build_pipeline(schema)

    record = Record(
        id="demo-001",
        source_text="INVOICE #4471  Date: 2026-03-14  Total: $1,240.00  From: Acme Freight Co.",
    )
    record = pipeline.run(record)

    print(f"Record {record.id}: {record.decision.action.value.upper()} "
          f"({record.decision.reason})\n")
    for name, f in record.fields.items():
        sigs = ", ".join(f"{s.name}={s.value:.2f}" for s in f.signals)
        print(f"  {name:16} value={f.value!r:22} risk={f.risk:.2f} "
              f"-> {f.decision.action.value:8} [{sigs}]")


if __name__ == "__main__":
    main()