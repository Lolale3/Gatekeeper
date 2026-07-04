"""gatekeeper.data -- the §1 dataset layer: labeled examples, a type-aware
correctness function, deterministic splits, and swappable dataset loaders."""
from .example import GroundTruth, LabeledExample
from .matching import is_correct
from .loader import DatasetLoader, Splits, split_examples
from .fixtures import FixtureInvoiceLoader, invoice_schema

__all__ = [
    "GroundTruth", "LabeledExample",
    "is_correct",
    "DatasetLoader", "Splits", "split_examples",
    "FixtureInvoiceLoader", "invoice_schema",
]