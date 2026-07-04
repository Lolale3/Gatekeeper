"""gatekeeper -- a confidence-gating / selective-escalation layer for LLM
extraction. §0: foundations & contracts. §1: dataset layer."""
from .types import Record, Field, Signal, Decision, Action
from .schema import Schema, FieldSpec
from .interfaces import Extractor, SignalGenerator, Calibrator, Policy
from .pipeline import Pipeline
from .config import PipelineConfig, build_pipeline

# §1 dataset layer
from .data import (
    GroundTruth, LabeledExample, is_correct,
    DatasetLoader, Splits, split_examples,
    FixtureInvoiceLoader, invoice_schema,
)

__all__ = [
    # §0
    "Record", "Field", "Signal", "Decision", "Action",
    "Schema", "FieldSpec",
    "Extractor", "SignalGenerator", "Calibrator", "Policy",
    "Pipeline", "PipelineConfig", "build_pipeline",
    # §1
    "GroundTruth", "LabeledExample", "is_correct",
    "DatasetLoader", "Splits", "split_examples",
    "FixtureInvoiceLoader", "invoice_schema",
]