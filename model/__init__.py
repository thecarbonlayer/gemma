"""The ``model/`` package — provider access + costing.

This is the "talking to a model" package: the model seam (``chat``), the provider
config (``Provider`` + presets), a deterministic ``fake`` provider for offline
tests, and the pricing/costing helpers. Add a model without touching the harness:
the harness depends only on ``chat`` and ``Provider``, not on any one provider's
implementation.
"""

from __future__ import annotations

from model.client import chat
from model.fake import FakeProvider, fake
from model.openai_compatible import complete_openai
from model.pricing import (
    PRICES,
    cost,
    cost_from_usage,
    format_cost,
    price_for,
)
from model.provider import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMResponse,
    OnDelta,
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
    "PRICES",
    "FakeProvider",
    "LLMResponse",
    "OnDelta",
    "Provider",
    "chat",
    "complete_openai",
    "cost",
    "cost_from_usage",
    "fake",
    "format_cost",
    "lmstudio",
    "ollama",
    "openai",
    "openrouter",
    "price_for",
]
