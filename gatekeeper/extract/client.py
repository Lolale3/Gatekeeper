"""LLM transport clients. A client knows how to send a prompt to one backend and
get text back -- nothing more. Splitting transport from the extractor's logic is
what makes backends swappable, and lets §3 reuse the same client for sampling.
"""
from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from typing import Union


class LLMClientError(RuntimeError):
    """Raised when a backend call fails (network, timeout, bad status)."""


class LLMClient(ABC):
    name: str = "llm_client"

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.0,
                 json_mode: bool = True) -> str:
        """Return the model's raw text response for a single prompt."""


class OllamaClient(LLMClient):
    """Talks to a local Ollama server over its REST API (default localhost:11434).
    Standard library only -- no extra dependencies."""
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434", timeout: float = 120.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.0,
                 json_mode: bool = True) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"        # Ollama constrains output to valid JSON
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as e:                 # network / timeout / bad status
            raise LLMClientError(f"Ollama request failed: {e}") from e
        return body.get("response", "")


class FakeClient(LLMClient):
    """Returns canned responses for deterministic, offline tests. Pass a single
    string (always returned) or a list (returned in order, then cycled -- handy
    for §3 self-consistency, which calls the client several times)."""
    name = "fake"

    def __init__(self, responses: Union[str, list[str]]):
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self._i = 0
        self.calls: list[str] = []             # prompts seen, for test assertions

    def generate(self, prompt: str, *, temperature: float = 0.0,
                 json_mode: bool = True) -> str:
        self.calls.append(prompt)
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return resp