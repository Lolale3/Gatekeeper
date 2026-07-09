"""Interactive cost-slider demo (optional UI). Drag the review cost and watch each
field's threshold, verdict, and the record decision update live -- making the point
that the escalate/approve line is *economic*, not arbitrary (threshold = review cost
/ error cost). No model runs here; the slider is the whole interaction.

By default it loads real gated records captured from an eval run
(demo_records.json, produced by:  real_invoice_eval.py --dump-records demo_records.json).
If that file isn't present, it falls back to a small curated set.

    pip install -e ".[ui]"
    streamlit run examples/streamlit_app.py
"""
import json
import os

import pandas as pd
import streamlit as st

from gatekeeper import Record, invoice_schema, CostAwarePolicy
from gatekeeper.types import Action

SCHEMA = invoice_schema()

CURATED = [
    {"id": "inv-201", "fields": {
        "invoice_number": {"value": "7789-A", "risk": 0.10},
        "invoice_date": {"value": "2026-03-03", "risk": 0.08},
        "total_amount": {"value": "875.50", "risk": 0.10},
        "vendor_name": {"value": "Cardinal Supply", "risk": 0.05}}},
    {"id": "inv-202", "fields": {
        "invoice_number": {"value": "INV-2025-0088", "risk": 0.15},
        "invoice_date": {"value": "2025-11-05", "risk": 0.12},
        "total_amount": {"value": "2500", "risk": 0.24},
        "vendor_name": {"value": "Blue Ridge Logistics", "risk": 0.07}}},
    {"id": "inv-203", "fields": {
        "invoice_number": {"value": "#4471", "risk": 0.30},
        "invoice_date": {"value": "2026-03-14", "risk": 0.10},
        "total_amount": {"value": "1240", "risk": 0.40},
        "vendor_name": {"value": "Acme Freight Co.", "risk": 0.20}}},
]


def load_records():
    if os.path.exists("demo_records.json"):
        with open("demo_records.json", encoding="utf-8") as fh:
            raw = json.load(fh)
        recs = [{"id": r["id"], "fields": {
            n: {"value": d.get("value"), "risk": d.get("risk"), "correct": d.get("correct")}
            for n, d in r["fields"].items()}} for r in raw]
        return recs, True
    return CURATED, False


def flip_point(rec):
    """Review cost at which this record starts auto-approving (max over critical
    fields of risk x error_cost) -- used to pick a spread for a lively slider."""
    pts = [(rec["fields"][s.name]["risk"] or 1.0) * s.error_cost
           for s in SCHEMA.fields if s.critical]
    return max(pts) if pts else 0.0


def select_spread(recs, k=6):
    if len(recs) <= k:
        return recs
    ranked = sorted(recs, key=flip_point)
    idx = sorted(set(round(i * (len(ranked) - 1) / (k - 1)) for i in range(k)))
    return [ranked[i] for i in idx]


def build_record(item):
    rec = Record(id=item["id"], source_text="(pre-extracted)")
    for spec in SCHEMA.fields:
        fd = item["fields"][spec.name]
        f = rec.set_value(spec.name, fd["value"])
        f.risk = fd["risk"] if fd["risk"] is not None else 1.0
    return rec


st.set_page_config(page_title="gatekeeper — cost slider", layout="centered")
st.title("gatekeeper")
st.markdown("#### The escalate / approve line is economic, not arbitrary")

records_raw, is_real = load_records()
records_raw = select_spread(records_raw, k=6)
st.caption(
    ("Real invoices your pipeline gated — actual model extractions and calibrated "
     "risks. " if is_real else "Curated example invoices. ")
    + "Each field is auto-approved only when its risk falls below a threshold set by "
      "economics:  threshold = review cost / error cost.  Drag the review cost and "
      "watch the verdicts change. Fields marked ⚑ are critical — a critical field "
      "escalating routes the whole record to a human.")

review_cost = st.slider(
    "Review cost  (relative to a field's error cost)", 0.1, 3.0, 1.0, 0.1,
    help="Cheap reviews -> escalate readily. Expensive reviews -> tolerate more risk.")
policy = CostAwarePolicy(review_cost=review_cost)

records = [build_record(item) for item in records_raw]
for rec in records:
    policy.decide(rec, SCHEMA)
approved = sum(1 for r in records if r.decision.action is Action.APPROVE)

c1, c2, c3 = st.columns(3)
c1.metric("documents", len(records))
c2.metric("auto-approved", approved)
c3.metric("escalated", len(records) - approved)
st.divider()


def _color(val):
    return ("background-color: #FCEBEB; color: #791F1F" if val == "escalate"
            else "background-color: #E1F5EE; color: #08504F")


for item, rec in zip(records_raw, records):
    verdict = rec.decision.action.value
    dot = "🟢" if verdict == "approve" else "🟠"
    st.markdown(f"**{dot} {rec.id}** — record: **{verdict.upper()}**")
    rows = []
    for spec in SCHEMA.fields:
        f, d = rec.fields[spec.name], rec.fields[spec.name].decision
        row = {
            "field": spec.name + ("  ⚑" if spec.critical else ""),
            "value": f.value,
            "risk": round(f.risk, 2),
            "threshold": round(d.threshold, 2),
            "verdict": d.action.value,
        }
        if is_real:
            row["actually"] = "✓ correct" if item["fields"][spec.name].get("correct") else "✗ wrong"
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df.style.map(_color, subset=["verdict"]), hide_index=True,
                 use_container_width=True)