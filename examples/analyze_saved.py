"""Re-analyze a saved eval (real_eval_risks.json) instantly -- no re-extraction.
Try different operating points against risks/errors you already computed.

    python examples/analyze_saved.py
"""
import json

from gatekeeper.evaluation import aurc, selective_error_at_coverage
from gatekeeper.calibration import expected_calibration_error
from gatekeeper.policy import provable_risk_controlled_threshold


def main():
    with open("real_eval_risks.json") as fh:
        d = json.load(fh)
    risks, errors = d["risks"], d["errors"]
    print(f"{len(errors)} fields, base error {sum(errors)/len(errors):.1%}")
    print(f"AURC {aurc(risks, errors):.3f}   ECE {expected_calibration_error(risks, errors):.3f}\n")
    print("provable operating points (>= 90% confidence):")
    for alpha in (0.10, 0.20, 0.25, 0.30, 0.40):
        t = provable_risk_controlled_threshold(risks, errors, alpha=alpha, delta=0.1)
        appr = [(r, e) for r, e in zip(risks, errors) if r <= t]
        cov = len(appr) / len(risks)
        act = (sum(e for _, e in appr) / len(appr)) if appr else 0.0
        print(f"  guarantee <= {alpha:.0%} error -> auto-approve {cov:.0%} (actual {act:.0%})")


if __name__ == "__main__":
    main()