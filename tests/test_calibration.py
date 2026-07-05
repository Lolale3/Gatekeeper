"""§4 milestone: the calibrator learns which signals predict errors, outputs valid
calibrated probabilities, the training-data builder labels errors correctly, and
ECE behaves (0 for honest predictions, large for overconfident ones)."""
from gatekeeper import Record, Signal, GroundTruth, invoice_schema
from gatekeeper.calibration import (
    LogisticCalibrator, build_training_data,
    expected_calibration_error, naive_risk,
)


def test_logistic_learns_that_low_rules_predicts_error():
    X, y = [], []
    for _ in range(20):
        X.append({"rules": 1.0, "grounding": 1.0}); y.append(0)   # clean -> correct
        X.append({"rules": 0.0, "grounding": 1.0}); y.append(1)   # rule fails -> error
    cal = LogisticCalibrator(lr=0.5, epochs=4000).fit(X, y)
    assert cal.weights_map()["rules"] < 0                          # higher rules -> lower risk
    assert (cal.predict_proba({"rules": 0.0, "grounding": 1.0})
            > cal.predict_proba({"rules": 1.0, "grounding": 1.0}))


def test_calibrate_populates_risk_in_unit_interval():
    schema = invoice_schema()
    cal = LogisticCalibrator().fit([{"rules": 1.0, "grounding": 1.0}], [0])
    rec = Record(id="t", source_text="x")
    for spec in schema.fields:
        f = rec.set_value(spec.name, "v")
        f.add_signal(Signal(name="rules", value=1.0, generator="rules"))
        f.add_signal(Signal(name="grounding", value=1.0, generator="grounding"))
    cal.calibrate(rec, schema)
    for spec in schema.fields:
        r = rec.fields[spec.name].risk
        assert r is not None and 0.0 <= r <= 1.0


def test_build_training_data_labels_errors():
    schema = invoice_schema()
    rec = Record(id="t", source_text="INVOICE 4471 Total 1240")
    rec.set_value("invoice_number", "4471")
    rec.set_value("invoice_date", None)
    rec.set_value("total_amount", "9999")          # wrong
    rec.set_value("vendor_name", None)
    gt = GroundTruth(values={"invoice_number": "4471", "invoice_date": None,
                             "total_amount": "1240", "vendor_name": None})
    X, y = build_training_data([rec], [gt], schema)
    assert y == [0, 0, 1, 0]                        # only total_amount is an error


def test_unfitted_calibrator_falls_back_to_naive():
    schema = invoice_schema()
    cal = LogisticCalibrator()                      # never fitted
    rec = Record(id="t", source_text="x")
    for spec in schema.fields:
        f = rec.set_value(spec.name, "v")
        f.add_signal(Signal(name="rules", value=(0.5 if spec.name == "invoice_number" else 1.0),
                            generator="rules"))
    cal.calibrate(rec, schema)
    assert abs(rec.fields["invoice_number"].risk - 0.5) < 1e-9   # 1 - 0.5


def test_ece_zero_for_perfect_and_large_for_overconfident():
    assert expected_calibration_error([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]) == 0.0
    over = expected_calibration_error([0.9, 0.9, 0.9, 0.9], [0, 0, 0, 1])
    assert over > 0.5


def test_naive_risk():
    assert abs(naive_risk({"a": 1.0, "b": 1.0}) - 0.0) < 1e-9
    assert abs(naive_risk({"a": 0.0, "b": 1.0}) - 0.5) < 1e-9