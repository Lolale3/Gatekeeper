"""gatekeeper.signals -- §3: signal generators. Each produces a per-field score in
[0,1] (higher = more confident); the calibrator (§4) fuses them into a risk."""
from .consistency import SelfConsistencySignal
from .grounding import GroundingSignal
from .rules import RuleSignal, dtype_valid

__all__ = ["SelfConsistencySignal", "GroundingSignal", "RuleSignal", "dtype_valid"]