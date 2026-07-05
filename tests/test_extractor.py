"""§2 milestone: prove the extractor's parsing/populating logic against a
FakeClient's canned responses -- clean JSON, fenced JSON, prose-wrapped JSON,
malformed output, extra keys -- with zero dependence on a live model."""
import pytest

from gatekeeper import Record, invoice_schema
from gatekeeper.extract import LLMExtractor, FakeClient, parse_json_object, locate_span

SOURCE = "INVOICE #4471  Date: 2026-03-14  Total: $1,240.00  From: Acme Freight Co."


def _run(fake_response, source=SOURCE):
    schema = invoice_schema()
    extractor = LLMExtractor(FakeClient(fake_response))
    record = extractor.extract(Record(id="t", source_text=source), schema)
    return record, schema


def test_clean_json_populates_all_fields():
    resp = ('{"invoice_number":"4471","invoice_date":"2026-03-14",'
            '"total_amount":"1240.00","vendor_name":"Acme Freight Co."}')
    rec, schema = _run(resp)
    assert set(rec.fields) == set(schema.field_names())
    assert rec.fields["invoice_number"].value == "4471"
    assert rec.fields["total_amount"].value == "1240.00"


def test_markdown_fenced_json_is_parsed():
    resp = '```json\n{"invoice_number":"4471","invoice_date":null,"total_amount":"1240.00","vendor_name":null}\n```'
    rec, _ = _run(resp)
    assert rec.fields["invoice_number"].value == "4471"
    assert rec.fields["invoice_date"].value is None          # JSON null -> None


def test_prose_wrapped_json_is_recovered():
    resp = 'Here is the extracted data:\n{"invoice_number":"4471","total_amount":"1240.00"}\nHope that helps!'
    rec, schema = _run(resp)
    assert rec.fields["invoice_number"].value == "4471"
    assert rec.fields["vendor_name"].value is None           # missing key -> None


def test_malformed_response_degrades_gracefully():
    rec, schema = _run("this is not json at all")
    assert set(rec.fields) == set(schema.field_names())      # no crash
    assert all(rec.fields[n].value is None for n in schema.field_names())
    assert rec.meta.get("extraction_error")                  # failure recorded


def test_extra_keys_are_ignored():
    resp = '{"invoice_number":"4471","total_amount":"1240.00","random_field":"ignore me"}'
    rec, schema = _run(resp)
    assert "random_field" not in rec.fields
    assert set(rec.fields) == set(schema.field_names())


def test_key_matching_is_tolerant():
    # spaces/case in the returned key still map to invoice_number
    resp = '{"Invoice Number":"4471","total_amount":"1240.00"}'
    rec, _ = _run(resp)
    assert rec.fields["invoice_number"].value == "4471"


def test_source_span_grounding():
    resp = ('{"invoice_number":"4471","invoice_date":null,'
            '"total_amount":"1240.00","vendor_name":"Acme Freight Co."}')
    rec, _ = _run(resp)
    assert rec.fields["invoice_number"].source_span is not None   # present in source
    assert rec.fields["invoice_date"].source_span is None         # null -> no span


def test_parse_json_object_variants():
    assert parse_json_object('{"a":1}') == {"a": 1}
    assert parse_json_object('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_json_object('prefix {"a":1} suffix') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_locate_span_is_case_insensitive():
    assert locate_span("From: ACME Freight Co.", "acme freight co.") == "ACME Freight Co."
    assert locate_span("nothing relevant", "missing") is None