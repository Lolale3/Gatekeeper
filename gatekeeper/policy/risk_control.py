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


# --- provable variant (conformal-style risk control) --------------------------
# The empirical threshold above is the "point-estimate" baseline that Trust-or-
# Escalate (Jung et al., 2024) shows can fail to hold on small calibration sets.
# Below we threshold on a (1 - delta) UPPER CONFIDENCE BOUND of the error rate
# instead of the point estimate, giving a genuine guarantee.
import math


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(1.0, total)


def binomial_upper_bound(k: int, n: int, delta: float) -> float:
    """(1 - delta) Clopper-Pearson upper confidence bound on the true rate given k
    'successes' (here: errors) in n trials. The largest p with P(Bin(n,p) <= k) >= delta.
    Wider (more conservative) when n is small -- you cannot certify a low rate from
    few samples."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):                       # bisection: CDF is decreasing in p
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) >= delta:
            lo = mid
        else:
            hi = mid
    return lo


def provable_risk_controlled_threshold(risks, errors, alpha: float,
                                       delta: float = 0.1) -> float:
    """Largest risk threshold t such that the (1 - delta) upper bound on the error
    rate among auto-approved items (risk <= t) is <= alpha. Guarantee: with
    probability >= 1 - delta, the true error rate among auto-approved items is <=
    alpha. Multiplicity across candidate thresholds is handled with a Bonferroni
    (union-bound) split of delta -- conservative but valid; fixed-sequence testing
    would be a tighter alternative."""
    paired = sorted(zip(risks, errors), key=lambda re: re[0])
    if not paired:
        return 0.0
    thresholds = sorted({r for r, _ in paired})
    delta_each = delta / len(thresholds)
    best_t = 0.0
    for t in thresholds:
        approved = [e for r, e in paired if r <= t]
        if binomial_upper_bound(sum(approved), len(approved), delta_each) <= alpha:
            best_t = max(best_t, t)
    return best_t