"""§5 milestone: the cost-aware policy derives per-field thresholds from economics,
a costlier field escalates at lower risk, only a critical field's escalation
escalates the record, the review-cost dial moves the decision, and the
risk-controlled threshold honors the error budget."""
from gatekeeper import Record, Schema, FieldSpec, GroundTruth, Action
from gatekeeper.policy import CostAwarePolicy, risk_controlled_threshold


def _rec(risks, schema):
    rec = Record(id="t", source_text="x")
    for spec in schema.fields:
        f = rec.set_value(spec.name, "v")
        f.risk = risks[spec.name]
    return rec


def test_field_threshold_from_costs():
    p = CostAwarePolicy(review_cost=1.0)
    assert p.field_threshold(5.0) == 0.2
    assert abs(p.field_threshold(3.0) - 1 / 3) < 1e-9
    assert p.field_threshold(1.0) == 1.0
    assert p.field_threshold(0.5) == 1.0            # clamped to 1.0
    assert p.field_threshold(0.0) == 1.0


def test_costlier_field_escalates_at_lower_risk():
    schema = Schema("x", [
        FieldSpec("rate", dtype="number", critical=True, error_cost=10.0),   # thr 0.10
        FieldSpec("ref", dtype="exact", critical=False, error_cost=1.0),      # thr 1.00
    ])
    rec = _rec({"rate": 0.3, "ref": 0.3}, schema)
    CostAwarePolicy(review_cost=1.0).decide(rec, schema)
    assert rec.fields["rate"].decision.action is Action.ESCALATE   # 0.30 > 0.10
    assert rec.fields["ref"].decision.action is Action.APPROVE     # 0.30 < 1.00


def test_record_escalates_only_on_critical_field():
    schema = Schema("x", [
        FieldSpec("crit", dtype="number", critical=True, error_cost=10.0),    # thr 0.10
        FieldSpec("noncrit", dtype="number", critical=False, error_cost=10.0),
    ])
    rec = _rec({"crit": 0.05, "noncrit": 0.5}, schema)
    CostAwarePolicy(review_cost=1.0).decide(rec, schema)
    assert rec.fields["noncrit"].decision.action is Action.ESCALATE
    assert rec.decision.action is Action.APPROVE                   # non-critical doesn't pull the record

    rec2 = _rec({"crit": 0.5, "noncrit": 0.05}, schema)
    CostAwarePolicy(review_cost=1.0).decide(rec2, schema)
    assert rec2.decision.action is Action.ESCALATE                 # critical does


def test_review_cost_dials_escalation():
    schema = Schema("x", [FieldSpec("f", dtype="number", critical=True, error_cost=5.0)])
    hi = _rec({"f": 0.3}, schema)
    CostAwarePolicy(review_cost=1.0).decide(hi, schema)            # thr 0.20 -> escalate
    assert hi.decision.action is Action.ESCALATE
    lo = _rec({"f": 0.3}, schema)
    CostAwarePolicy(review_cost=3.0).decide(lo, schema)           # thr 0.60 -> approve
    assert lo.decision.action is Action.APPROVE


def test_risk_controlled_threshold_honors_budget():
    schema = Schema("doc", [FieldSpec("x", dtype="exact", critical=True)])
    risks = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    errs  = [0,    0,    0,   1,   1,   1,   1,   1]
    records, gts = [], []
    for r, e in zip(risks, errs):
        rec = Record(id="t", source_text="x")
        f = rec.set_value("x", "wrong" if e else "ok"); f.risk = r
        records.append(rec); gts.append(GroundTruth(values={"x": "ok"}))

    t = risk_controlled_threshold(records, gts, schema, target_error=0.0)
    approved = [(r, e) for r, e in zip(risks, errs) if r <= t]
    assert all(e == 0 for _, e in approved)          # zero-error budget honored
    assert t >= 0.1                                  # includes the clean low-risk items

    t2 = risk_controlled_threshold(records, gts, schema, target_error=0.25)
    approved2 = [e for r, e in zip(risks, errs) if r <= t2]
    assert sum(approved2) / len(approved2) <= 0.25   # 25% budget honored
    assert t2 >= t                                   # a looser budget approves at least as much