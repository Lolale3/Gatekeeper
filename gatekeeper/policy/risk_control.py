"""Risk-controlled threshold selection: instead of minimizing expected cost, pick
the single global risk threshold that keeps the error rate among AUTO-APPROVED
items within a target budget, measured on labeled calibration data.

This is the empirical version of a coverage guarantee: choose the largest
threshold (max auto-approval coverage) whose approved set has an empirical error
rate <= target_error. Fit on the calibration split (disjoint from test), it gives
an honest operating point; a finite-sample correction would turn the empirical
bound into a formal one. Use the returned threshold with a global-threshold policy.
"""
from __future__ import annotations

from ..schema import Schema
from ..data.matching import is_correct


def risk_controlled_threshold(records, ground_truths, schema: Schema,
                              target_error: float, *, only_critical: bool = False) -> float:
    items = []                              # (risk, is_error) over calibration fields
    for record, gt in zip(records, ground_truths):
        for spec in schema.fields:
            if only_critical and not spec.critical:
                continue
            field = record.fields[spec.name]
            if field.risk is None:
                continue
            err = 0 if is_correct(field.value, gt.get(spec.name), spec.dtype) else 1
            items.append((field.risk, err))
    if not items:
        return 0.0

    candidates = sorted({r for r, _ in items} | {0.0, 1.0})
    best_t = 0.0
    for t in candidates:
        approved = [err for r, err in items if r <= t]
        err_rate = (sum(approved) / len(approved)) if approved else 0.0
        if err_rate <= target_error:
            best_t = max(best_t, t)         # largest satisfying threshold = max coverage
    return best_t