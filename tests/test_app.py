"""§8 milestone: the report builds the right structure, HTML renders, the CLI runs
end-to-end offline, and a fitted calibrator survives a save/load round-trip."""
import json

from gatekeeper import Record, Signal, invoice_schema, CostAwarePolicy
from gatekeeper.calibration import LogisticCalibrator
from gatekeeper.app import build_report, render_text, render_html, save_calibrator, load_calibrator
from gatekeeper.app.cli import main


def _decided_record(risks):
    schema = invoice_schema()
    rec = Record(id="doc-1", source_text="x")
    for spec in schema.fields:
        f = rec.set_value(spec.name, "v")
        f.risk = risks[spec.name]
    CostAwarePolicy(review_cost=1.0).decide(rec, schema)
    return rec, schema


def test_report_structure_and_queue():
    # total_amount (cost 5 -> thr 0.20) risk 0.6 escalates; others approve
    rec, schema = _decided_record(
        {"invoice_number": 0.05, "invoice_date": 0.05, "total_amount": 0.6, "vendor_name": 0.05})
    report = build_report([rec], schema)
    assert report["summary"]["total"] == 1
    assert report["summary"]["escalated"] == 1        # record escalated (critical field)
    assert any(q["field"] == "total_amount" for q in report["queue"])
    assert "escalate" in render_text(report)


def test_render_html_is_standalone_and_contains_content():
    rec, schema = _decided_record(
        {"invoice_number": 0.05, "invoice_date": 0.05, "total_amount": 0.6, "vendor_name": 0.05})
    htmldoc = render_html(build_report([rec], schema))
    assert htmldoc.strip().startswith("<!DOCTYPE html>")
    assert "doc-1" in htmldoc and "total_amount" in htmldoc


def test_calibrator_save_load_roundtrip(tmp_path):
    cal = LogisticCalibrator().fit([{"g": 1.0}, {"g": 0.0}], [0, 1])
    path = tmp_path / "cal.json"
    save_calibrator(cal, str(path))
    loaded = load_calibrator(str(path))
    assert loaded.feature_names == cal.feature_names
    assert abs(loaded.predict_proba({"g": 0.0}) - cal.predict_proba({"g": 0.0})) < 1e-9


def test_cli_runs_offline(tmp_path, capsys):
    html_path = tmp_path / "report.html"
    rc = main(["--offline", "--html", str(html_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "record:" in out and "summary:" in out
    assert html_path.exists()
    assert html_path.read_text().startswith("<!DOCTYPE html>")