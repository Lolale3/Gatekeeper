"""The risk-coverage curve and its summaries -- the headline measurement. Given a
calibrated risk and a ground-truth error indicator per test item, sweep the
approval threshold and measure, at each coverage level, the error rate among
auto-approved items. Everything here is computed on the TEST split only; fitting
the calibrator on the same data would make these numbers meaningless.
"""
from __future__ import annotations

import math

from ..schema import Schema
from ..data.matching import is_correct


def collect_risk_labels(records, ground_truths, schema: Schema, *, only_critical: bool = False):
    """From evaluated test records, gather (risks, errors) per field."""
    risks, errors = [], []
    for record, gt in zip(records, ground_truths):
        for spec in schema.fields:
            if only_critical and not spec.critical:
                continue
            field = record.fields[spec.name]
            if field.risk is None:
                continue
            risks.append(field.risk)
            errors.append(0 if is_correct(field.value, gt.get(spec.name), spec.dtype) else 1)
    return risks, errors


def risk_coverage_curve(risks, errors):
    """Sweep the approval threshold; return [(coverage, error_rate)] from low to
    full coverage, approving the lowest-risk items first."""
    paired = sorted(zip(risks, errors), key=lambda t: t[0])   # ascending risk
    n = len(paired)
    if n == 0:
        return []
    points, cum_err = [], 0
    for k, (_, err) in enumerate(paired, start=1):
        cum_err += err
        points.append((k / n, cum_err / k))
    return points


def selective_error_at_coverage(risks, errors, coverage: float) -> float:
    """Error rate among auto-approved items when approving `coverage` fraction of
    the lowest-risk items."""
    paired = sorted(zip(risks, errors), key=lambda t: t[0])
    n = len(paired)
    if n == 0:
        return 0.0
    k = max(1, min(n, math.ceil(coverage * n)))
    chosen = paired[:k]
    return sum(e for _, e in chosen) / k


def aurc(risks, errors) -> float:
    """Area under the risk-coverage curve (trapezoidal, normalized by covered
    span). Lower is better -- a well-ordered system keeps error low even at high
    coverage."""
    points = risk_coverage_curve(risks, errors)
    if len(points) < 2:
        return points[0][1] if points else 0.0
    area = 0.0
    for (c0, e0), (c1, e1) in zip(points, points[1:]):
        area += (c1 - c0) * (e0 + e1) / 2.0
    span = points[-1][0] - points[0][0]
    return area / span if span > 0 else points[-1][1]