"""The provider configuration object — the model seam, made explicit.

Every model call in the whole course goes through ``chat()`` (see
``model/client.py``). The agent never talks to a provider directly, so swapping
providers (Ch14) is just changing the ``LLM_BASE_URL`` / ``LLM_MODEL`` env vars —
nothing above this file changes.

Targets any OpenAI-compatible ``/chat/completions`` endpoint:
  - local dev / acceptance: LM Studio (defaults below)
  - CI / hosted: OpenRouter, OpenAI, Ollama, ...

A ``Provider`` is pure config (base_url / model / api_key). It can also carry an
optional ``responder`` — a callable that produces an ``LLMResponse`` directly —
so a fake provider can inject deterministic, offline responses without any HTTP.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "http://192.168.189.144:1234/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b"
DEFAULT_API_KEY = "lm-studio"


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Real env vars always win."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None = None
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class Provider:
    """The model seam, made explicit (ch-14). Any OpenAI-compatible endpoint.

    Swap the provider and nothing above this object changes.

    ``responder`` is an optional escape hatch: when set, ``chat()`` calls it
    instead of hitting the network. This is how the ``fake`` provider injects
    deterministic responses for offline tests. ``from_env`` always leaves it
    ``None`` — env-configured providers go over HTTP.
    """

    base_url: str
    model: str
    api_key: str = "x"
    responder: Callable[..., LLMResponse] | None = None

    @classmethod
    def from_env(cls) -> Provider:
        _load_dotenv()
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("LLM_API_KEY", DEFAULT_API_KEY),
        )


# Presets — same agent, one line to switch.
def lmstudio(model: str = DEFAULT_MODEL) -> Provider:
    return Provider(DEFAULT_BASE_URL, model, "lm-studio")


def openrouter(model: str, api_key: str) -> Provider:
    return Provider("https://openrouter.ai/api/v1", model, api_key)


def ollama(model: str) -> Provider:
    return Provider("http://localhost:11434/v1", model, "ollama")


def openai(model: str, api_key: str) -> Provider:
    return Provider("https://api.openai.com/v1", model, api_key)
