"""Policy demo (offline, deterministic). Shows two things:

  1. The same fixed risks produce different verdicts as the review cost changes --
     cheap reviews escalate more, expensive reviews escalate less -- because the
     per-field threshold is review_cost / error_cost.
  2. The risk-controlled threshold: pick the operating point that keeps the error
     rate among auto-approved items within a target budget, from labeled data.

    PYTHONPATH=. python examples/policy_demo.py
"""
from gatekeeper import Record, Schema, FieldSpec, GroundTruth, invoice_schema
from gatekeeper.policy import CostAwarePolicy, risk_controlled_threshold

# fixed calibrated risks for one record (as if produced by §4)
RISKS = {"invoice_number": 0.40, "invoice_date": 0.30, "total_amount": 0.15, "vendor_name": 0.05}


def _record(risks):
    schema = invoice_schema()
    rec = Record(id="demo", source_text="(one invoice)")
    for spec in schema.fields:
        f = rec.set_value(spec.name, "v")
        f.risk = risks[spec.name]
    return rec, schema


def show(review_cost):
    rec, schema = _record(RISKS)
    CostAwarePolicy(review_cost=review_cost).decide(rec, schema)
    print(f"\nreview_cost = {review_cost:g}   =>   record: {rec.decision.action.value.upper()}")
    for spec in schema.fields:
        f = rec.fields[spec.name]
        d = f.decision
        crit = "critical" if spec.critical else "        "
        print(f"  {spec.name:15} risk={f.risk:.2f}  thr={d.threshold:.2f}  "
              f"(cost {spec.error_cost:g}) {crit}  -> {d.action.value}")


def risk_controlled_part():
    print("\n" + "=" * 60)
    print("risk-controlled threshold (pick operating point for an error budget)")
    schema = Schema("doc", [FieldSpec("x", dtype="exact", critical=True)])
    risks = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
    errs  = [0,    0,    0,    0,    1,   0,   1,    1,    1,    1]
    records, gts = [], []
    for r, e in zip(risks, errs):
        rec = Record(id="c", source_text="x")
        f = rec.set_value("x", "wrong" if e else "ok"); f.risk = r
        records.append(rec); gts.append(GroundTruth(values={"x": "ok"}))
    for target in (0.0, 0.1, 0.25):
        t = risk_controlled_threshold(records, gts, schema, target_error=target)
        approved = [(r, e) for r, e in zip(risks, errs) if r <= t]
        cov = len(approved) / len(risks)
        err = (sum(e for _, e in approved) / len(approved)) if approved else 0.0
        print(f"  target error <= {target:.2f}  ->  threshold {t:.2f}  "
              f"(auto-approves {cov:.0%}, actual error {err:.0%})")


def main():
    print("same risks, different review costs -- watch the record verdict flip:")
    for rc in (0.2, 1.0, 3.0):
        show(rc)
    risk_controlled_part()


if __name__ == "__main__":
    main()