"""Config-driven wiring. In later sections, which extractor / signals / policy
to use is selected here from config, so experiments (§7) become a config change,
not a code change. For §0 this just builds the default walking-skeleton pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

from .schema import Schema
from .pipeline import Pipeline
from .stubs import (
    StubExtractor,
    ConstantSignalGenerator,
    ConstantCalibrator,
    ThresholdPolicy,
)


@dataclass
class PipelineConfig:
    extractor: str = "stub"
    signal_generators: list[str] = dc_field(default_factory=lambda: ["constant"])
    calibrator: str = "constant"
    policy: str = "threshold"
    threshold: float = 0.5


def build_pipeline(schema: Schema, config: Optional[PipelineConfig] = None) -> Pipeline:
    config = config or PipelineConfig()
    # In §2+ this becomes a registry lookup keyed on the config strings above.
    return Pipeline(
        schema=schema,
        extractor=StubExtractor(),
        signal_generators=[ConstantSignalGenerator()],
        calibrator=ConstantCalibrator(),
        policy=ThresholdPolicy(threshold=config.threshold),
    )