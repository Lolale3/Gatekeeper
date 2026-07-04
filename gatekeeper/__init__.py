"""gatekeeper -- a confidence-gating / selective-escalation layer for LLM
extraction. §0: foundations, contracts, and a walking skeleton."""
from .types import Record, Field, Signal, Decision, Action
from .schema import Schema, FieldSpec
from .interfaces import Extractor, SignalGenerator, Calibrator, Policy
from .pipeline import Pipeline
from .config import PipelineConfig, build_pipeline

__all__ = [
    "Record", "Field", "Signal", "Decision", "Action",
    "Schema", "FieldSpec",
    "Extractor", "SignalGenerator", "Calibrator", "Policy",
    "Pipeline", "PipelineConfig", "build_pipeline",
]