"""§1b milestone: the synthetic freight generator produces schema-conformant labeled
emails with valid-typed gold, is deterministic, and its business rules behave."""
from gatekeeper.data.freight import (
    SyntheticFreightLoader, freight_schema, EQUIPMENT,
    _positive_rate, _plausible_weight, _known_equipment, _place_ok,
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
    assert _plausible_weight("42") is False            # '42k' misread as 42 -> caught
    assert _plausible_weight("999999") is False
    assert _known_equipment("reefer") is True
    assert _known_equipment("spaceship") is False


def test_place_rule_catches_phrases_and_misses():
    assert _place_ok("Ontario") is True
    assert _place_ok("Memphis, TN") is True
    assert _place_ok("Inland Empire area (specifically Ontario)") is False   # phrase
    assert _place_ok("the greater metro region of somewhere") is False       # too long
    assert _place_ok(None) is False                                          # mandatory field missed


def test_confidence_layer_catches_freight_errors():
    """The hardened freight signals + calibration escalate most injected errors."""
    import random
    from gatekeeper import Record
    from gatekeeper.signals import GroundingSignal, RuleSignal
    from gatekeeper.calibration import LogisticCalibrator, build_training_data
    from gatekeeper.policy import CostAwarePolicy
    from gatekeeper.data import split_examples, is_correct
    from gatekeeper.types import Action

    schema = freight_schema()

    def build(examples, seed):
        rng = random.Random(seed)
        recs = []
        for ex in examples:
            g = ex.ground_truth.values
            v = dict(g)
            if rng.random() < 0.30:
                v["origin"] = f"{g['origin']} area (specifically {g['origin']})"
            elif rng.random() < 0.20:
                v["origin"] = None
            if rng.random() < 0.30 and g["weight"]:
                v["weight"] = str(int(g["weight"]) // 1000)
            rec = Record(id=ex.id, source_text=ex.source_text)
            for spec in schema.fields:
                rec.set_value(spec.name, v.get(spec.name))
            for gen in (GroundingSignal(), RuleSignal()):
                gen.generate(rec, schema)
            recs.append(rec)
        return recs, [ex.ground_truth for ex in examples]

    splits = split_examples(SyntheticFreightLoader(n=60).load())
    cal, cgt = build(splits.calibration, 1)
    X, y = build_training_data(cal, cgt, schema)
    calibrator = LogisticCalibrator().fit(X, y)

    test, tgt = build(splits.test, 2)
    policy = CostAwarePolicy(review_cost=1.0)
    caught = errors = 0
    for rec, gt in zip(test, tgt):
        calibrator.calibrate(rec, schema)
        policy.decide(rec, schema)
        for spec in schema.fields:
            f = rec.fields[spec.name]
            if not is_correct(f.value, gt.get(spec.name), spec.dtype):
                errors += 1
                caught += f.decision.action is Action.ESCALATE
    assert errors > 0
    assert caught / errors >= 0.8          # escalate the large majority of errors