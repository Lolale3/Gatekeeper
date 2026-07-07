"""gatekeeper -- a confidence-gating / selective-escalation layer for LLM
extraction. §0 foundations · §1 dataset · §2 extractor · §3 signals · §4 calibration · §5 policy · §6 feedback · §7 evaluation."""
from .types import Record, Field, Signal, Decision, Action
from .schema import Schema, FieldSpec
from .interfaces import Extractor, SignalGenerator, Calibrator, Policy
from .pipeline import Pipeline
from .config import PipelineConfig, build_pipeline

from .data import (
    GroundTruth, LabeledExample, is_correct,
    DatasetLoader, Splits, split_examples,
    FixtureInvoiceLoader, invoice_schema,
    SyntheticFreightLoader, freight_schema,
    RealInvoiceLoader,
)
from .extract import LLMClient, OllamaClient, FakeClient, LLMExtractor
from .signals import SelfConsistencySignal, GroundingSignal, RuleSignal
from .calibration import (
    LogisticCalibrator, build_training_data, naive_risk,
    reliability_table, expected_calibration_error,
)
from .policy import CostAwarePolicy, risk_controlled_threshold
from .evaluation import (
    collect_risk_labels, risk_coverage_curve, selective_error_at_coverage,
    aurc, ablate_signals,
)
from .feedback import FeedbackLoop, corrections_to_training_data, select_escalated_records

__all__ = [
    "Record", "Field", "Signal", "Decision", "Action",
    "Schema", "FieldSpec",
    "Extractor", "SignalGenerator", "Calibrator", "Policy",
    "Pipeline", "PipelineConfig", "build_pipeline",
    "GroundTruth", "LabeledExample", "is_correct",
    "DatasetLoader", "Splits", "split_examples",
    "FixtureInvoiceLoader", "invoice_schema",
    "SyntheticFreightLoader", "freight_schema",
    "RealInvoiceLoader",
    "LLMClient", "OllamaClient", "FakeClient", "LLMExtractor",
    "SelfConsistencySignal", "GroundingSignal", "RuleSignal",
    "LogisticCalibrator", "build_training_data", "naive_risk",
    "reliability_table", "expected_calibration_error",
    "CostAwarePolicy", "risk_controlled_threshold",
    "collect_risk_labels", "risk_coverage_curve", "selective_error_at_coverage",
    "aurc", "ablate_signals",
    "FeedbackLoop", "corrections_to_training_data", "select_escalated_records",
]