"""§1 milestone: prove the answer key loads, the correctness function is
type-aware and handles absence, and the splits are deterministic and disjoint."""
from gatekeeper import FixtureInvoiceLoader, invoice_schema, split_examples, is_correct


def test_fixtures_load_and_conform_to_schema():
    schema = invoice_schema()
    examples = FixtureInvoiceLoader().load()
    assert len(examples) >= 10
    names = set(schema.field_names())
    for ex in examples:
        assert ex.source_text.strip()
        assert set(ex.ground_truth.values).issubset(names)


def test_correctness_number_ignores_formatting():
    assert is_correct("$1,240.00", "1240.00", "number") is True
    assert is_correct("1240", "1250", "number") is False


def test_correctness_date_ignores_format():
    assert is_correct("14/03/2026", "2026-03-14", "date") is True
    assert is_correct("2026-03-15", "2026-03-14", "date") is False


def test_correctness_exact_rejects_near_miss():
    assert is_correct("4471", "4471", "exact") is True
    assert is_correct("4471", "4472", "exact") is False


def test_correctness_str_normalizes():
    assert is_correct("ACME FREIGHT CO.", "Acme Freight Co.", "str") is True


def test_absent_field_cases():
    assert is_correct(None, None, "str") is True          # both absent -> correct
    assert is_correct("", None, "str") is True            # empty == absent
    assert is_correct("Acme", None, "str") is False       # hallucinated a value
    assert is_correct(None, "Acme", "str") is False       # missed a real value


def test_split_is_deterministic_and_disjoint():
    examples = FixtureInvoiceLoader().load()
    s1 = split_examples(examples, seed=13)
    s2 = split_examples(examples, seed=13)
    order1 = [e.id for e in s1.dev + s1.calibration + s1.test]
    order2 = [e.id for e in s2.dev + s2.calibration + s2.test]
    assert order1 == order2                                # reproducible

    all_ids = {e.id for e in examples}
    got = ({e.id for e in s1.dev} | {e.id for e in s1.calibration}
           | {e.id for e in s1.test})
    assert got == all_ids                                  # complete
    assert {e.id for e in s1.calibration}.isdisjoint({e.id for e in s1.test})  # no leakage