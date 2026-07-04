"""The §0 milestone: prove the contracts fit and a Record traverses the whole
pipeline, arriving with a value, at least one signal, a risk, and a decision on
every field, plus a record-level decision."""
from gatekeeper import Record, Schema, FieldSpec, PipelineConfig, build_pipeline
from gatekeeper import Action


def _schema() -> Schema:
    return Schema(
        doc_type="invoice",
        fields=[
            FieldSpec("invoice_number", critical=True),
            FieldSpec("total_amount", dtype="number", critical=True),
            FieldSpec("vendor_name", critical=False),
        ],
    )


def test_record_traverses_pipeline():
    schema = _schema()
    pipeline = build_pipeline(schema)
    record = pipeline.run(Record(id="t1", source_text="some invoice text"))

    assert set(record.fields) == set(schema.field_names())
    for name in schema.field_names():
        f = record.fields[name]
        assert f.value is not None
        assert f.signals, f"{name} has no signals"
        assert f.risk is not None
        assert f.decision is not None
        assert f.decision.action in (Action.APPROVE, Action.ESCALATE)

    assert record.decision is not None
    assert record.decision.action in (Action.APPROVE, Action.ESCALATE)


def test_high_threshold_approves_everything():
    schema = _schema()
    pipeline = build_pipeline(schema, PipelineConfig(threshold=1.0))
    record = pipeline.run(Record(id="t2", source_text="x"))
    assert record.decision.action is Action.APPROVE
    assert all(f.decision.action is Action.APPROVE for f in record.fields.values())