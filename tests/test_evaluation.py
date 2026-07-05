"""§7 milestone: the risk-coverage math is correct, AURC rewards good risk
ordering, selective error reads off the right value, and the ablation flags a
useful signal as mattering more than a noise signal."""
from gatekeeper.evaluation import (
    risk_coverage_curve, selective_error_at_coverage, aurc, ablate_signals,
)


def test_risk_coverage_curve_math():
    pts = risk_coverage_curve([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
    assert [c for c, _ in pts] == [0.25, 0.5, 0.75, 1.0]
    assert pts[0][1] == 0.0                       # lowest-risk item is correct
    assert abs(pts[-1][1] - 0.5) < 1e-9           # full coverage = overall error rate


def test_selective_error_at_coverage():
    risks, errors = [0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]
    assert selective_error_at_coverage(risks, errors, 0.5) == 0.0      # two safest are correct
    assert abs(selective_error_at_coverage(risks, errors, 1.0) - 0.5) < 1e-9


def test_aurc_rewards_good_ordering():
    good = aurc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])   # errors on the riskiest items
    bad = aurc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])    # errors on the safest items
    assert good < bad


def test_ablation_flags_useful_over_noise():
    Xc, yc = [], []
    for i in range(20):
        Xc.append({"useful": 1.0, "noise": float(i % 2)}); yc.append(0)
        Xc.append({"useful": 0.0, "noise": float(i % 2)}); yc.append(1)
    res = ablate_signals(Xc, yc, list(Xc), list(yc), ["useful", "noise"])
    assert res["without_useful"] > res["without_noise"]   # removing the useful signal hurts more
    assert res["without_useful"] >= res["full"]