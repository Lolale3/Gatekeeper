"""Offline, deterministic proof that the confidence layer CATCHES realistic freight
extraction errors. We simulate a real model's output on the freight emails -- mostly
correct, but with the characteristic mistakes (a phrase instead of a city, '42k'
misread as 42, a dropped field, an off rate) -- then run the full confidence
pipeline and show those errors getting ESCALATED instead of waved through.

    python examples/freight_gate_demo.py
"""
import random

from gatekeeper import Record
from gatekeeper.data.freight import SyntheticFreightLoader, freight_schema
from gatekeeper.data import split_examples, is_correct
from gatekeeper.signals import GroundingSignal, RuleSignal
from gatekeeper.calibration import LogisticCalibrator, build_training_data
from gatekeeper.policy import CostAwarePolicy
from gatekeeper.types import Action


def simulate(ex, rng):
    """A plausible model extraction: gold values, with characteristic errors injected."""
    g = ex.ground_truth.values
    v = dict(g)
    if rng.random() < 0.30:                                  # grabbed the metro phrase
        v["origin"] = f"{g['origin']} area (specifically {g['origin']})"
    elif rng.random() < 0.20:                                # dropped the origin entirely
        v["origin"] = None
    if rng.random() < 0.30 and g["weight"]:                  # '42k' misread as 42
        v["weight"] = str(int(g["weight"]) // 1000)
    if rng.random() < 0.15:                                  # rate off by a bit
        v["rate"] = str(int(g["rate"]) - 250)
    return v


def make_records(examples, rng):
    schema = freight_schema()
    records = []
    for ex in examples:
        rec = Record(id=ex.id, source_text=ex.source_text)
        for spec in schema.fields:
            rec.set_value(spec.name, simulate(ex, rng).get(spec.name))
        for gen in (GroundingSignal(), RuleSignal()):
            gen.generate(rec, schema)
        records.append(rec)
    return records, [ex.ground_truth for ex in examples]


def main() -> None:
    schema = freight_schema()
    splits = split_examples(SyntheticFreightLoader(n=60).load())

    cal_recs, cal_gts = make_records(splits.calibration, random.Random(1))
    X, y = build_training_data(cal_recs, cal_gts, schema)
    calibrator = LogisticCalibrator().fit(X, y)
    print(f"fit on {len(X)} calibration fields ({sum(y)} errors)\n")

    test_recs, test_gts = make_records(splits.test, random.Random(2))
    policy = CostAwarePolicy(review_cost=1.0)
    for rec in test_recs:
        calibrator.calibrate(rec, schema)
        policy.decide(rec, schema)

    # did we catch the errors? measure recall on errors vs false-alarms on correct fields
    caught = total_err = approved_ok = total_ok = 0
    for rec, gt in zip(test_recs, test_gts):
        for spec in schema.fields:
            f = rec.fields[spec.name]
            wrong = not is_correct(f.value, gt.get(spec.name), spec.dtype)
            escalated = f.decision.action is Action.ESCALATE
            if wrong:
                total_err += 1; caught += escalated
            else:
                total_ok += 1; approved_ok += (not escalated)
    print(f"errors caught (escalated):   {caught}/{total_err}")
    print(f"correct fields auto-approved: {approved_ok}/{total_ok}")

    esc_records = sum(1 for r in test_recs if r.decision.action is Action.ESCALATE)
    print(f"records escalated to a human: {esc_records}/{len(test_recs)}\n")

    print("sample gated fields:")
    for rec in test_recs[:3]:
        print(f"  {rec.id}: {rec.decision.action.value}")
        for spec in schema.fields:
            f = rec.fields[spec.name]
            print(f"    [{f.decision.action.value:8}] {spec.name:11} risk={f.risk:.2f}  value={f.value!r}")


if __name__ == "__main__":
    main()