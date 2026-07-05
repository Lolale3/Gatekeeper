"""gatekeeper.evaluation -- §7: the eval harness. Risk-coverage curve, AURC,
selective error, per-signal ablation, and (optional) matplotlib plots. All
measured on the held-out test split."""
from .curves import (
    collect_risk_labels, risk_coverage_curve, selective_error_at_coverage, aurc,
)
from .ablation import ablate_signals
from .plots import plot_risk_coverage, plot_reliability

__all__ = [
    "collect_risk_labels", "risk_coverage_curve", "selective_error_at_coverage",
    "aurc", "ablate_signals", "plot_risk_coverage", "plot_reliability",
]