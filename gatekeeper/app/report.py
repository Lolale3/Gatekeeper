"""Turn processed records into a report structure, then render it as text (for the
CLI) or as a standalone, dependency-free HTML page (for a shareable artifact)."""
from __future__ import annotations

import html

from ..schema import Schema


def build_report(records, schema: Schema) -> dict:
    docs, queue = [], []
    for rec in records:
        fields = []
        for spec in schema.fields:
            f = rec.fields[spec.name]
            d = f.decision
            action = d.action.value if d else "?"
            fields.append({
                "name": spec.name, "value": f.value,
                "risk": f.risk, "threshold": d.threshold if d else None,
                "action": action, "critical": spec.critical,
            })
            if action == "escalate":
                queue.append({"doc": rec.id, "field": spec.name, "risk": f.risk})
        docs.append({
            "id": rec.id,
            "action": rec.decision.action.value if rec.decision else "?",
            "fields": fields,
        })
    approved = sum(1 for d in docs if d["action"] == "approve")
    summary = {"total": len(docs), "auto_approved": approved,
               "escalated": len(docs) - approved, "queue_size": len(queue)}
    return {"docs": docs, "queue": queue, "summary": summary}


def _fmt_risk(r):
    return f"{r:.2f}" if isinstance(r, (int, float)) else "-"


def render_text(report: dict) -> str:
    out = []
    for doc in report["docs"]:
        out.append(f"\n{doc['id']}  ->  record: {doc['action'].upper()}")
        for f in doc["fields"]:
            crit = " (critical)" if f["critical"] else ""
            out.append(f"  [{f['action']:8}] {f['name']:15} "
                       f"risk={_fmt_risk(f['risk'])} thr={_fmt_risk(f['threshold'])}"
                       f"  value={f['value']!r}{crit}")
    s = report["summary"]
    out.append(f"\nsummary: {s['auto_approved']}/{s['total']} auto-approved, "
               f"{s['escalated']} escalated, {s['queue_size']} field(s) in the review queue")
    if report["queue"]:
        out.append("review queue (highest risk first):")
        for q in sorted(report["queue"], key=lambda x: -(x["risk"] or 0)):
            out.append(f"  {q['doc']} · {q['field']}  risk={_fmt_risk(q['risk'])}")
    return "\n".join(out)


def _chip(action):
    color = "#0F6E56" if action == "approve" else "#854F0B"
    bg = "#E1F5EE" if action == "approve" else "#FAEEDA"
    return (f'<span style="background:{bg};color:{color};font-size:12px;'
            f'padding:3px 10px;border-radius:8px">{html.escape(action)}</span>')


def render_html(report: dict, title: str = "gatekeeper report") -> str:
    cards = []
    for doc in report["docs"]:
        rec_chip = _chip("approve" if doc["action"] == "approve" else "escalate")
        rows = []
        for f in doc["fields"]:
            crit = ('<span style="color:#854F0B;font-size:11px"> · critical</span>'
                    if f["critical"] else "")
            rows.append(
                f'<tr style="border-top:0.5px solid #e7e5dc">'
                f'<td style="padding:8px 0;color:#5f5e5a">{html.escape(f["name"])}{crit}</td>'
                f'<td style="font-family:ui-monospace,monospace">{html.escape(str(f["value"]))}</td>'
                f'<td style="text-align:right">{_fmt_risk(f["risk"])}</td>'
                f'<td style="text-align:right">{_chip(f["action"])}</td></tr>')
        cards.append(f"""
  <div class="card">
    <div class="head"><div><div class="docid">{html.escape(doc['id'])}</div>
      <div class="sub">{len(doc['fields'])} fields</div></div>{rec_chip}</div>
    <table><tr class="th"><th>field</th><th>value</th><th style="text-align:right">risk</th>
      <th style="text-align:right">verdict</th></tr>{''.join(rows)}</table>
  </div>""")
    s = report["summary"]
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:2rem auto;
padding:0 1rem;color:#1a1a1a;background:#faf9f5;line-height:1.6}}
h1{{font-size:20px;font-weight:500}}
.summary{{color:#5f5e5a;font-size:14px;margin-bottom:1.5rem}}
.card{{background:#fff;border:0.5px solid #e7e5dc;border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem}}
.head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}}
.docid{{font-weight:500;font-size:15px}} .sub{{font-size:13px;color:#8a887f}}
table{{width:100%;font-size:13px;border-collapse:collapse}} .th{{color:#8a887f}}
.th th{{font-weight:400;text-align:left;padding-bottom:4px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="summary">{s['auto_approved']} of {s['total']} documents auto-approved · {s['escalated']} escalated · {s['queue_size']} field(s) in the review queue</p>
{''.join(cards)}
</body></html>"""