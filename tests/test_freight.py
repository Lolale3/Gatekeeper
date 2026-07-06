"""§1b milestone: the synthetic freight generator produces schema-conformant labeled
emails with valid-typed gold, is deterministic, and its business rules behave."""
from gatekeeper.data.freight import (
    SyntheticFreightLoader, freight_schema, EQUIPMENT,
    _positive_rate, _plausible_weight, _known_equipment,
)
from gatekeeper.data.matching import to_number, to_date


def test_examples_conform_to_schema():
    schema = freight_schema()
    names = set(schema.field_names())
    examples = SyntheticFreightLoader(n=8).load()
    assert len(examples) == 8
    for ex in examples:
        assert ex.source_text.strip()
        assert set(ex.ground_truth.values) == names          # every field labeled
        assert to_number(ex.ground_truth.get("rate")) is not None
        assert to_date(ex.ground_truth.get("pickup_date")) is not None
        assert ex.ground_truth.get("equipment") in EQUIPMENT


def test_generation_is_deterministic():
    a = [e.source_text for e in SyntheticFreightLoader(n=6, seed=7).load()]
    b = [e.source_text for e in SyntheticFreightLoader(n=6, seed=7).load()]
    assert a == b


def test_freight_schema_critical_fields():
    crit = set(freight_schema().critical_fields())
    assert crit == {"origin", "destination", "rate"}


def test_freight_rules():
    assert _positive_rate("2400") is True
    assert _positive_rate("0") is False
    assert _plausible_weight("42000") is True
    assert _plausible_weight("999999") is False
    assert _known_equipment("reefer") is True
    assert _known_equipment("spaceship") is False