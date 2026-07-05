"""gatekeeper.extract -- §2: the real extractor. Model-agnostic extraction logic
over swappable transport clients (Ollama local; Fake for offline tests)."""
from .client import LLMClient, OllamaClient, FakeClient, LLMClientError
from .extractor import LLMExtractor, build_prompt, parse_json_object, locate_span

__all__ = [
    "LLMClient", "OllamaClient", "FakeClient", "LLMClientError",
    "LLMExtractor", "build_prompt", "parse_json_object", "locate_span",
]