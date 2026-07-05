"""Config-driven wiring. Which extractor / signals / policy to use is selected
here, so experiments (§7) become a config change, not a code change. §3 adds real
signal generators; the self-consistency generator needs a sampling extractor
(built with a temperature > 0), so it is only wired when Ollama is the backend.
Components can also be injected directly via the `extractor=` / `signal_generators=`
overrides (used by tests and demos, which run offline against the FakeClient).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

from .schema import Schema
from .pipeline import Pipeline
from .interfaces import Extractor, SignalGenerator, Calibrator
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
    consistency_n: int = 5
    consistency_temperature: float = 0.7
    calibrator: str = "constant"
    policy: str = "threshold"
    threshold: float = 0.5


def _build_extractor(config: PipelineConfig) -> Extractor:
    if config.extractor == "ollama":
        from .extract import LLMExtractor, OllamaClient
        return LLMExtractor(OllamaClient(model=config.ollama_model, host=config.ollama_host))
    return StubExtractor()


def _build_signal_generators(config: PipelineConfig) -> list[SignalGenerator]:
    gens: list[SignalGenerator] = []
    for name in config.signal_generators:
        if name == "grounding":
            from .signals import GroundingSignal
            gens.append(GroundingSignal())
        elif name == "rules":
            from .signals import RuleSignal
            gens.append(RuleSignal())
        elif name == "self_consistency":
            from .signals import SelfConsistencySignal
            from .extract import LLMExtractor, OllamaClient
            sampler = LLMExtractor(
                OllamaClient(model=config.ollama_model, host=config.ollama_host),
                temperature=config.consistency_temperature,
            )
            gens.append(SelfConsistencySignal(sampler, n=config.consistency_n))
        elif name == "constant":
            gens.append(ConstantSignalGenerator())
    return gens or [ConstantSignalGenerator()]


def _build_calibrator(config: PipelineConfig) -> Calibrator:
    if config.calibrator == "logistic":
        from .calibration import LogisticCalibrator
        return LogisticCalibrator()          # fit separately on the calibration split
    return ConstantCalibrator()


def build_pipeline(schema: Schema, config: Optional[PipelineConfig] = None, *,
                   extractor: Optional[Extractor] = None,
                   signal_generators: Optional[list[SignalGenerator]] = None,
                   calibrator: Optional[Calibrator] = None) -> Pipeline:
    config = config or PipelineConfig()
    return Pipeline(
        schema=schema,
        extractor=extractor or _build_extractor(config),
        signal_generators=signal_generators or _build_signal_generators(config),
        calibrator=calibrator or _build_calibrator(config),
        policy=ThresholdPolicy(threshold=config.threshold),
    )