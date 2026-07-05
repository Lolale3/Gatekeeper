"""Feedback-loop demo (offline, deterministic). Shows two things:

  1. Before/after: fold human-corrected escalations into the calibration data,
     refit, and measure the risk-coverage curve on a FROZEN test set -- it improves.
  2. Why escalated labels are worth more: escalation concentrates errors, so the
     same label budget spent on escalated items surfaces far more errors (and thus
     far more information) than spending it on random items -- active learning.

    PYTHONPATH=. python examples/feedback_demo.py
"""
from gatekeeper.calibration import LogisticCalibrator, expected_calibration_error
from gatekeeper.feedback import FeedbackLoop


def at_threshold(risks, labels, t):
    """Decision quality at a fixed risk threshold: auto-approve risk <= t."""
    approved = [(r, y) for r, y in zip(risks, labels) if r <= t]
    if not approved:
        return 0.0, 0.0
    coverage = len(approved) / len(risks)
    error = sum(y for _, y in approved) / len(approved)
    return coverage, error


import random

_noise = random.Random(1)


def item(a, b):
    # correct only when both real signals are high; 'c' is pure noise that a tiny
    # base set will spuriously over-fit -- feedback washes it out.
    return {"a": float(a), "b": float(b), "c": float(_noise.random() > 0.5)}, (0 if (a and b) else 1)


def dataset(spec):
    out = []
    for (a, b), n in spec:
        out += [item(a, b) for _ in range(n)]
    return [x for x, _ in out], [y for _, y in out]


def main() -> None:
    # base calibration: tiny and biased -- only ever saw a,b move together, so it
    # mis-weights them and mis-ranks the "one high, one low" cases on test.
    base_X, base_y = dataset([((1, 1), 8), ((0, 0), 4)])

    # incoming production stream: mostly clean, with the informative error cases rare
    pool_X, pool_y = dataset([((1, 1), 40), ((1, 0), 6), ((0, 1), 6), ((0, 0), 8)])

    # frozen held-out test: balanced across all four cases
    test_X, test_y = dataset([((1, 1), 15), ((1, 0), 15), ((0, 1), 15), ((0, 0), 15)])

    base_cal = LogisticCalibrator().fit(base_X, base_y)
    risks_before = [base_cal.predict_proba(x) for x in test_X]
    cov_b, err_b = at_threshold(risks_before, test_y, 0.5)
    print("BEFORE feedback (calibrator fit on tiny base set):")
    print(f"  ECE {expected_calibration_error(risks_before, test_y):.3f}   "
          f"at risk<=0.5: auto-approve {cov_b:.0%}, error among approved {err_b:.0%}")

    # the system escalates its highest-risk pool items; a human corrects them
    pool_risk = [base_cal.predict_proba(x) for x in pool_X]
    order = sorted(range(len(pool_X)), key=lambda i: pool_risk[i], reverse=True)
    budget = 20
    escalated = order[:budget]

    loop = FeedbackLoop(base_X, base_y)
    loop.add_labeled([pool_X[i] for i in escalated], [pool_y[i] for i in escalated])
    fed_cal = loop.fit()
    risks_after = [fed_cal.predict_proba(x) for x in test_X]
    cov_a, err_a = at_threshold(risks_after, test_y, 0.5)
    print(f"\nAFTER feedback (+{loop.feedback_count} corrected escalations):")
    print(f"  ECE {expected_calibration_error(risks_after, test_y):.3f}   "
          f"at risk<=0.5: auto-approve {cov_a:.0%}, error among approved {err_a:.0%}")
    print("  (better-calibrated risk -> the fixed threshold now escalates the errors "
          "it used to wave through)")

    # why escalated labels are worth more: error concentration vs random
    import random
    rng = random.Random(0)
    rand = rng.sample(range(len(pool_X)), budget)
    esc_err = sum(pool_y[i] for i in escalated) / budget
    rnd_err = sum(pool_y[i] for i in rand) / budget
    print("\nwhy escalated labels are worth more (same 20-label budget):")
    print(f"  escalated selection: {esc_err:.0%} of labels are actual errors")
    print(f"  random selection   : {rnd_err:.0%} of labels are actual errors")


if __name__ == "__main__":
    main()