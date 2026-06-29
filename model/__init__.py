"""The ``model/`` package — provider access.

This is the "talking to a model" package: the model seam (``chat``), the provider
config (``Provider`` + presets), and a deterministic ``fake`` provider for offline
tests. Add a model without touching the harness: the harness depends only on
``chat`` and ``Provider``, not on any one provider's implementation.
"""

from __future__ import annotations

from model.client import chat
from model.fake import FakeProvider, fake
from model.openai_compatible import complete_openai
from model.provider import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMResponse,
    Provider,
    lmstudio,
    ollama,
    openai,
    openrouter,
)

__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "FakeProvider",
    "LLMResponse",
    "Provider",
    "chat",
    "complete_openai",
    "fake",
    "lmstudio",
    "ollama",
    "openai",
    "openrouter",
]
