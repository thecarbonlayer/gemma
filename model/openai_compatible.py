"""The OpenAI-compatible HTTP call.

One implementation of the model seam: POST to ``/chat/completions`` on any
OpenAI-compatible endpoint (LM Studio, OpenRouter, Ollama, OpenAI, ...). This is
the body that used to live inside ``chat()``; ``model/client.py`` now dispatches
here when a provider has no ``responder``.

Going fully provider-agnostic (an Anthropic-native call, Gemini, ...) is "drop a
file next to this one" — the harness never changes because it depends only on the
``chat`` seam, not on this function.
"""

from __future__ import annotations

import httpx

from model.provider import LLMResponse, Provider


def complete_openai(
    provider: Provider,
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 180.0,
) -> LLMResponse:
    """One call to an OpenAI-compatible model, through a provider.

    ``max_tokens`` is generous on purpose: Gemma is a reasoning model and spends
    tokens thinking (``reasoning_content``) before it produces visible content.
    """
    base_url = provider.base_url.rstrip("/")
    model = model or provider.model
    api_key = provider.api_key

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    resp = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    msg = choice.get("message", {})
    return LLMResponse(
        content=msg.get("content") or "",
        reasoning=msg.get("reasoning_content"),
        tool_calls=msg.get("tool_calls") or [],
        usage=data.get("usage", {}),
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )
