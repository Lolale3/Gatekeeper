"""Eval-harness demo (offline, deterministic). Builds a synthetic calibration and
test set with one useful signal (predicts errors) and one noise signal (doesn't),
fits the calibrator, and reports the headline measurements on the TEST split:
risk-coverage read-offs, AURC, ECE, and the per-signal ablation. If matplotlib is
installed, it also saves the risk-coverage and reliability plots.

    PYTHONPATH=. python examples/eval_demo.py
"""
from gatekeeper.calibration import LogisticCalibrator, expected_calibration_error
from gatekeeper.evaluation import (
    risk_coverage_curve, selective_error_at_coverage, aurc, ablate_signals,
)


import random

_rng = random.Random(0)


def block(useful, n, err_count):
    # n items at this 'useful' level; the first err_count are errors. 'noise' is
    # drawn independently, so it carries no information about the errors.
    return [({"useful": useful, "noise": float(_rng.random() > 0.5)}, 1 if i < err_count else 0)
            for i in range(n)]


def main() -> None:
    cal = block(1.0, 20, 1) + block(0.5, 20, 8) + block(0.0, 20, 18)
    test = block(1.0, 20, 2) + block(0.5, 20, 7) + block(0.0, 20, 17)
    Xc = [x for x, _ in cal]; yc = [y for _, y in cal]
    Xt = [x for x, _ in test]; yt = [y for _, y in test]

    cal_model = LogisticCalibrator().fit(Xc, yc)
    risks = [cal_model.predict_proba(x) for x in Xt]

    print(f"test set: {len(yt)} items, {sum(yt)} errors ({sum(yt)/len(yt):.0%})\n")
    print("risk-coverage read-offs (error rate among auto-approved):")
    for cov in (0.5, 0.7, 0.9, 1.0):
        print(f"  at {cov:.0%} coverage -> {selective_error_at_coverage(risks, yt, cov):.1%} error")
    print(f"\nAURC (lower = better ordering): {aurc(risks, yt):.3f}")
    print(f"ECE  (lower = better calibrated): {expected_calibration_error(risks, yt):.3f}\n")

    print("per-signal ablation (AURC when a signal is removed; higher = signal mattered):")
    res = ablate_signals(Xc, yc, Xt, yt, ["useful", "noise"])
    print(f"  full model        : {res['full']:.3f}")
    print(f"  without 'useful'  : {res['without_useful']:.3f}   "
          f"(+{res['without_useful'] - res['full']:.3f})")
    print(f"  without 'noise'   : {res['without_noise']:.3f}   "
          f"(+{res['without_noise'] - res['full']:.3f})")

    try:
        from gatekeeper.evaluation import plot_risk_coverage, plot_reliability
        p1 = plot_risk_coverage(risks, yt, "eval_risk_coverage.png")
        p2 = plot_reliability(risks, yt, "eval_reliability.png")
        print(f"\nplots saved: {p1}, {p2}")
    except ImportError:
        print("\n(install matplotlib to also save plots: pip install 'gatekeeper[eval]')")


if __name__ == "__main__":
    main()