"""The DatasetLoader contract and deterministic, leakage-free splitting.

Fixtures, real invoices and (later) synthetic freight are all implementations of
DatasetLoader, each yielding the same LabeledExample shape. Splitting is seeded
so runs reproduce, and calibration/test are kept disjoint -- calibrating and
evaluating on the same examples would make the risk-coverage numbers (§7) a lie.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .example import LabeledExample


class DatasetLoader(ABC):
    name: str = "dataset"

    @abstractmethod
    def load(self) -> list[LabeledExample]: ...


@dataclass
class Splits:
    dev: list[LabeledExample]           # hand-iteration: rules, prompts, thresholds
    calibration: list[LabeledExample]   # fit the calibrator (§4)
    test: list[LabeledExample]          # held out; eval harness only (§7)

    def summary(self) -> str:
        total = len(self.dev) + len(self.calibration) + len(self.test)
        return (f"dev={len(self.dev)}  calibration={len(self.calibration)}  "
                f"test={len(self.test)}  total={total}")


def split_examples(examples, *, dev: float = 0.2, calibration: float = 0.4,
                   test: float = 0.4, seed: int = 13) -> Splits:
    total_frac = dev + calibration + test
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total_frac}")
    items = list(examples)
    random.Random(seed).shuffle(items)          # deterministic given the seed
    n = len(items)
    n_dev = round(n * dev)
    n_cal = round(n * calibration)
    return Splits(
        dev=items[:n_dev],
        calibration=items[n_dev:n_dev + n_cal],
        test=items[n_dev + n_cal:],             # remainder -> always sums to n
    )