"""Calibration demo (offline, deterministic). Builds a small labeled set, runs the
cheap signals (grounding + rules), fits the pure-Python logistic calibrator, and
shows: the learned signal weights, the ECE improvement over a naive baseline, and
the calibrated risk for a clean vs a flagged field.

Tiny data -> illustrative, not statistically robust. The real calibration comes
from fitting on a real dataset; here we prove the machinery and the measurement.

    PYTHONPATH=. python examples/calibrate_demo.py
"""
from gatekeeper import Record, GroundTruth, invoice_schema
from gatekeeper.signals import GroundingSignal, RuleSignal
from gatekeeper.calibration import (
    LogisticCalibrator, build_training_data,
    expected_calibration_error, naive_risk, reliability_table,
)

# (source, extracted values, gold). A deliberate spread of error types:
#   * grounded-but-wrong: '#4471' extracted where the source really shows '#4471'
#     -- grounding stays 1.0 (blind), only rules flags the stray '#'
#   * ungrounded: an invented vendor -- grounding flags it, rules is blind
#   * both: a negative total -- both signals flag it
# so neither signal alone separates all errors, and the calibrator must weight both.
def _src(num, tot, ven):
    return f"INVOICE {num}  Date: 2026-03-14  Total: {tot}  From: {ven}"

EXAMPLES = [
    # clean
    (_src("4471", "$1,240.00", "Acme Freight Co."),
     {"invoice_number": "4471", "invoice_date": "2026-03-14", "total_amount": "1240", "vendor_name": "Acme Freight Co."},
     {"invoice_number": "4471", "invoice_date": "2026-03-14", "total_amount": "1240", "vendor_name": "Acme Freight Co."}),
    # grounded-but-wrong: '#4471' is really in the source -> grounding blind, rules flags
    (_src("#4471", "$1,240.00", "Acme Freight Co."),
     {"invoice_number": "#4471", "invoice_date": "2026-03-14", "total_amount": "1240", "vendor_name": "Acme Freight Co."},
     {"invoice_number": "4471", "invoice_date": "2026-03-14", "total_amount": "1240", "vendor_name": "Acme Freight Co."}),
    # invented vendor -> grounding flags, rules blind
    (_src("8080", "$2,340.75", "Ironclad Freightways"),
     {"invoice_number": "8080", "invoice_date": "2026-03-14", "total_amount": "2340.75", "vendor_name": "Ghost Logistics"},
     {"invoice_number": "8080", "invoice_date": "2026-03-14", "total_amount": "2340.75", "vendor_name": "Ironclad Freightways"}),
    # clean
    (_src("33219", "$9,875.25", "Parkway Logistics Group"),
     {"invoice_number": "33219", "invoice_date": "2026-03-14", "total_amount": "9875.25", "vendor_name": "Parkway Logistics Group"},
     {"invoice_number": "33219", "invoice_date": "2026-03-14", "total_amount": "9875.25", "vendor_name": "Parkway Logistics Group"}),
    # grounded-but-wrong '#55'
    (_src("#55", "$640.00", "Northwind Carriers"),
     {"invoice_number": "#55", "invoice_date": "2026-03-14", "total_amount": "640", "vendor_name": "Northwind Carriers"},
     {"invoice_number": "55", "invoice_date": "2026-03-14", "total_amount": "640", "vendor_name": "Northwind Carriers"}),
    # negative total -> both signals flag it
    (_src("6620", "$480.00", "Harbor Point Shipping"),
     {"invoice_number": "6620", "invoice_date": "2026-03-14", "total_amount": "-50", "vendor_name": "Harbor Point Shipping"},
     {"invoice_number": "6620", "invoice_date": "2026-03-14", "total_amount": "480", "vendor_name": "Harbor Point Shipping"}),
    # clean
    (_src("7789", "$875.50", "Cardinal Supply"),
     {"invoice_number": "7789", "invoice_date": "2026-03-14", "total_amount": "875.50", "vendor_name": "Cardinal Supply"},
     {"invoice_number": "7789", "invoice_date": "2026-03-14", "total_amount": "875.50", "vendor_name": "Cardinal Supply"}),
    # invented vendor
    (_src("100234", "$12,000.00", "Blue Ridge Logistics"),
     {"invoice_number": "100234", "invoice_date": "2026-03-14", "total_amount": "12000", "vendor_name": "Phantom Corp"},
     {"invoice_number": "100234", "invoice_date": "2026-03-14", "total_amount": "12000", "vendor_name": "Blue Ridge Logistics"}),
]


def labeled_records():
    schema = invoice_schema()
    records, gts = [], []
    for source, values, gold in EXAMPLES:
        rec = Record(id="demo", source_text=source)
        for spec in schema.fields:
            rec.set_value(spec.name, values.get(spec.name))
        for gen in (GroundingSignal(), RuleSignal()):
            gen.generate(rec, schema)
        records.append(rec)
        gts.append(GroundTruth(values=gold))
    return records, gts, schema


def main() -> None:
    records, gts, schema = labeled_records()
    X, y = build_training_data(records, gts, schema)
    print(f"{len(X)} field-instances, {sum(y)} of them errors.\n")

    cal = LogisticCalibrator().fit(X, y)
    print("learned signal weights (more negative = signal more strongly predicts 'correct'):")
    for name, w in cal.weights_map().items():
        print(f"  {name:12} {w:+.2f}")
    print(f"  bias         {cal.bias:+.2f}\n")

    calibrated = [cal.predict_proba(v) for v in X]
    naive = [naive_risk(v) for v in X]
    print(f"ECE  naive (1 - mean signals): {expected_calibration_error(naive, y):.3f}")
    print(f"ECE  calibrated (logistic)   : {expected_calibration_error(calibrated, y):.3f}")
    print("  (lower = better calibrated; tiny illustrative dataset)\n")

    clean = {"grounding": 1.0, "rules": 1.0}
    flagged = {"grounding": 1.0, "rules": 0.5}      # rules flagged it (e.g. the '#')
    print(f"risk of a clean field   (grounding=1.0, rules=1.0): {cal.predict_proba(clean):.2f}")
    print(f"risk of a flagged field (grounding=1.0, rules=0.5): {cal.predict_proba(flagged):.2f}")


if __name__ == "__main__":
    main()