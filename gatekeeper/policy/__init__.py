"""gatekeeper.policy -- §5: turn calibrated risk into approve/escalate decisions.
Cost-aware per-field thresholds, plus a risk-controlled threshold for a target
error budget."""
from .cost_aware import CostAwarePolicy
from .risk_control import (
    risk_controlled_threshold, provable_risk_controlled_threshold,
    binomial_upper_bound,
)

__all__ = ["CostAwarePolicy", "risk_controlled_threshold",
           "provable_risk_controlled_threshold", "binomial_upper_bound"]