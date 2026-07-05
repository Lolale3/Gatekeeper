"""Config-driven wiring. Which extractor / signals / policy to use is selected
here, so experiments (§7) become a config change, not a code change. §2 adds the
'ollama' extractor option; a Fake- or Ollama-backed extractor can also be passed
in directly via the `extractor=` override (used by tests and demos).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

from .schema import Schema
from .pipeline import Pipeline
from .interfaces import Extractor
from .stubs import (
    StubExtractor,
    ConstantSignalGenerator,
    ConstantCalibrator,
    ThresholdPolicy,
)


@dataclass
class PipelineConfig:
    extractor: str = "stub"                 # "stub" | "ollama"
    ollama_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    signal_generators: list[str] = dc_field(default_factory=lambda: ["constant"])
    calibrator: str = "constant"
    policy: str = "threshold"
    threshold: float = 0.5


def _build_extractor(config: PipelineConfig) -> Extractor:
    if config.extractor == "ollama":
        from .extract import LLMExtractor, OllamaClient      # lazy: only if selected
        return LLMExtractor(OllamaClient(model=config.ollama_model, host=config.ollama_host))
    return StubExtractor()


def build_pipeline(schema: Schema, config: Optional[PipelineConfig] = None,
                   *, extractor: Optional[Extractor] = None) -> Pipeline:
    config = config or PipelineConfig()
    return Pipeline(
        schema=schema,
        extractor=extractor or _build_extractor(config),
        signal_generators=[ConstantSignalGenerator()],
        calibrator=ConstantCalibrator(),
        policy=ThresholdPolicy(threshold=config.threshold),
    )