"""gatekeeper -- a confidence-gating / selective-escalation layer for LLM
extraction. §0 foundations · §1 dataset · §2 extractor · §3 signals · §4 calibration."""
from .types import Record, Field, Signal, Decision, Action
from .schema import Schema, FieldSpec
from .interfaces import Extractor, SignalGenerator, Calibrator, Policy
from .pipeline import Pipeline
from .config import PipelineConfig, build_pipeline

from .data import (
    GroundTruth, LabeledExample, is_correct,
    DatasetLoader, Splits, split_examples,
    FixtureInvoiceLoader, invoice_schema,
)
from .extract import LLMClient, OllamaClient, FakeClient, LLMExtractor
from .signals import SelfConsistencySignal, GroundingSignal, RuleSignal
from .calibration import (
    LogisticCalibrator, build_training_data, naive_risk,
    reliability_table, expected_calibration_error,
)

__all__ = [
    "Record", "Field", "Signal", "Decision", "Action",
    "Schema", "FieldSpec",
    "Extractor", "SignalGenerator", "Calibrator", "Policy",
    "Pipeline", "PipelineConfig", "build_pipeline",
    "GroundTruth", "LabeledExample", "is_correct",
    "DatasetLoader", "Splits", "split_examples",
    "FixtureInvoiceLoader", "invoice_schema",
    "LLMClient", "OllamaClient", "FakeClient", "LLMExtractor",
    "SelfConsistencySignal", "GroundingSignal", "RuleSignal",
    "LogisticCalibrator", "build_training_data", "naive_risk",
    "reliability_table", "expected_calibration_error",
]