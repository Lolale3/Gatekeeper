"""§6 milestone: corrections become correct labels (only for fields the human
actually labeled), the loop accumulates and refits, informative feedback improves
calibration on a held-out set, and escalated records are selectable."""
from gatekeeper import Record, Schema, FieldSpec, GroundTruth, Signal, Decision, Action
from gatekeeper.feedback import (
    FeedbackLoop, corrections_to_training_data, select_escalated_records,
)
from gatekeeper.calibration import LogisticCalibrator, expected_calibration_error


def _schema():
    return Schema("inv", [
        FieldSpec("num", dtype="exact", critical=True),
        FieldSpec("total", dtype="number", critical=True),
    ])


def _record_with_signals(values, schema):
    rec = Record(id="r", source_text="x")
    for spec in schema.fields:
        f = rec.set_value(spec.name, values.get(spec.name))
        f.add_signal(Signal(name="rules", value=1.0, generator="rules"))
    return rec


def test_corrections_label_errors():
    schema = _schema()
    rec = _record_with_signals({"num": "#4471", "total": "1240"}, schema)   # num wrong
    corr = GroundTruth(values={"num": "4471", "total": "1240"})
    X, y = corrections_to_training_data([rec], [corr], schema)
    assert len(X) == 2
    assert y == [1, 0]                       # num is an error, total is correct


def test_corrections_only_use_labeled_fields():
    schema = _schema()
    rec = _record_with_signals({"num": "v", "total": "1"}, schema)
    corr = GroundTruth(values={"num": "v"})  # human labeled only 'num'
    X, y = corrections_to_training_data([rec], [corr], schema)
    assert len(X) == 1                        # only the labeled field becomes an instance


def test_feedback_loop_accumulates_and_refits():
    loop = FeedbackLoop([{"g": 1.0}], [0])
    assert loop.feedback_count == 0
    loop.add_labeled([{"g": 0.0}, {"g": 0.0}], [1, 1])
    assert loop.feedback_count == 2
    X, y = loop.training_data()
    assert len(X) == 3 and len(y) == 3
    cal = loop.fit()
    assert cal.predict_proba({"g": 0.0}) > cal.predict_proba({"g": 1.0})


def test_feedback_improves_calibration():
    def item(a, b):
        return {"a": float(a), "b": float(b)}, (0 if (a and b) else 1)

    def ds(spec):
        xs, ys = [], []
        for (a, b), n in spec:
            for _ in range(n):
                x, yy = item(a, b); xs.append(x); ys.append(yy)
        return xs, ys

    base_X, base_y = ds([((1, 1), 8), ((0, 0), 4)])            # biased tiny base
    fb_X, fb_y = ds([((1, 0), 6), ((0, 1), 6), ((0, 0), 4)])   # informative corrections
    test_X, test_y = ds([((1, 1), 10), ((1, 0), 10), ((0, 1), 10), ((0, 0), 10)])

    base = LogisticCalibrator().fit(base_X, base_y)
    ece_before = expected_calibration_error([base.predict_proba(x) for x in test_X], test_y)

    loop = FeedbackLoop(base_X, base_y)
    loop.add_labeled(fb_X, fb_y)
    fed = loop.fit()
    ece_after = expected_calibration_error([fed.predict_proba(x) for x in test_X], test_y)

    assert ece_after < ece_before


def test_select_escalated_records():
    r1 = Record(id="1", source_text="x"); r1.decision = Decision(action=Action.ESCALATE)
    r2 = Record(id="2", source_text="x"); r2.decision = Decision(action=Action.APPROVE)
    r3 = Record(id="3", source_text="x")                       # never decided
    assert [r.id for r in select_escalated_records([r1, r2, r3])] == ["1"]