"""§3 milestone: the three signals behave, and (crucially) they fail in different
places -- rules catch what grounding is blind to, and grounding is normalization-
aware so it stops false-alarming on correct normalized values."""
from gatekeeper import invoice_schema, Record
from gatekeeper.extract import LLMExtractor, FakeClient
from gatekeeper.signals import SelfConsistencySignal, GroundingSignal, RuleSignal, dtype_valid

SOURCE = "INVOICE #4471  Date: March 3, 2026  Total: $1,240.00  From: Acme Freight Co."


def _record_with(values, source=SOURCE):
    schema = invoice_schema()
    rec = Record(id="t", source_text=source)
    for spec in schema.fields:
        rec.set_value(spec.name, values.get(spec.name))
    return rec, schema


# --- grounding: normalization-aware ---

def test_grounding_normalized_number_is_grounded():
    rec, schema = _record_with({"total_amount": "1240"})     # source has $1,240.00
    GroundingSignal().generate(rec, schema)
    assert rec.fields["total_amount"].get_signal("grounding").value == 1.0


def test_grounding_date_across_formats():
    rec, schema = _record_with({"invoice_date": "2026-03-03"})  # source says "March 3, 2026"
    GroundingSignal().generate(rec, schema)
    assert rec.fields["invoice_date"].get_signal("grounding").value == 1.0


def test_grounding_flags_invented_value():
    rec, schema = _record_with({"vendor_name": "Nonexistent Corp"})
    GroundingSignal().generate(rec, schema)
    assert rec.fields["vendor_name"].get_signal("grounding").value == 0.0


def test_grounding_none_value_is_ok():
    rec, schema = _record_with({"vendor_name": None})
    GroundingSignal().generate(rec, schema)
    assert rec.fields["vendor_name"].get_signal("grounding").value == 1.0


def test_grounding_is_blind_to_leading_hash():
    rec, schema = _record_with({"invoice_number": "#4471"})   # '#4471' IS in the source
    GroundingSignal().generate(rec, schema)
    assert rec.fields["invoice_number"].get_signal("grounding").value == 1.0


# --- rules: catch what grounding misses ---

def test_rules_flag_leading_punctuation():
    rec, schema = _record_with({"invoice_number": "#4471"})
    RuleSignal().generate(rec, schema)                        # 2 checks, 1 fails -> 0.5
    assert rec.fields["invoice_number"].get_signal("rules").value == 0.5


def test_rules_pass_clean_identifier():
    rec, schema = _record_with({"invoice_number": "4471"})
    RuleSignal().generate(rec, schema)
    assert rec.fields["invoice_number"].get_signal("rules").value == 1.0


def test_rules_flag_nonpositive_amount():
    rec, schema = _record_with({"total_amount": "0"})
    RuleSignal().generate(rec, schema)
    assert rec.fields["total_amount"].get_signal("rules").value == 0.5


def test_dtype_valid():
    assert dtype_valid("1240", "number") is True
    assert dtype_valid("not a number", "number") is False
    assert dtype_valid("2026-03-14", "date") is True
    assert dtype_valid(None, "number") is True


# --- self-consistency: agreement judged with normalization ---

def test_self_consistency_full_agreement():
    schema = invoice_schema()
    resp = '{"invoice_number":"4471","total_amount":"1240"}'
    primary = LLMExtractor(FakeClient(resp)).extract(Record(id="t", source_text=SOURCE), schema)
    SelfConsistencySignal(LLMExtractor(FakeClient([resp] * 4)), n=4).generate(primary, schema)
    assert primary.fields["total_amount"].get_signal("self_consistency").value == 1.0


def test_self_consistency_partial_agreement_ignores_formatting():
    schema = invoice_schema()
    primary = LLMExtractor(FakeClient('{"total_amount":"1240"}')).extract(
        Record(id="t", source_text=SOURCE), schema)
    # 3 of 4 agree (1240 / 1240.00 both match; 1250 does not)
    samples = ['{"total_amount":"1240"}', '{"total_amount":"1240.00"}',
               '{"total_amount":"1250"}', '{"total_amount":"1240"}']
    SelfConsistencySignal(LLMExtractor(FakeClient(samples)), n=4).generate(primary, schema)
    assert primary.fields["total_amount"].get_signal("self_consistency").value == 0.75


def test_two_temperature_flags_unstable_fields():
    """A field whose value changes between low and high temperature is flagged
    unstable (0.0); a field that stays the same is stable (1.0)."""
    from gatekeeper import invoice_schema, Record
    from gatekeeper.extract import LLMExtractor, FakeClient
    from gatekeeper.signals import TwoTemperatureSignal

    schema = invoice_schema()
    low = LLMExtractor(FakeClient(
        '{"invoice_number":"111","invoice_date":"2026-01-01","total_amount":"100","vendor_name":"Acme"}'))
    high = LLMExtractor(FakeClient(
        '{"invoice_number":"999","invoice_date":"2026-01-01","total_amount":"100","vendor_name":"Acme"}'))

    rec = low.extract(Record(id="t", source_text="x"), schema)   # populate fields
    TwoTemperatureSignal(low, high, n=1).generate(rec, schema)

    def sig(field):
        return next(s.value for s in rec.fields[field].signals if s.name == "two_temperature")

    assert sig("invoice_number") == 0.0      # 111 (low) vs 999 (high) -> unstable
    assert sig("invoice_date") == 1.0        # unchanged -> stable
    assert sig("vendor_name") == 1.0