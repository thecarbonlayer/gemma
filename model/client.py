"""The model seam — the free ``chat()`` entry point.

Every model call in the whole course goes through this function. It defaults to
the env-configured provider and dispatches:
  - to the provider's ``responder`` when one is set (e.g. the fake provider), or
  - to the OpenAI-compatible HTTP call otherwise.

Keeping ``chat`` a free, module-level function is deliberate: the agent imports it
as a name (``from model import chat``) so tests can ``patch.object(mod, "chat")``.
"""

from __future__ import annotations

from model.openai_compatible import complete_openai
from model.provider import LLMResponse, Provider


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 180.0,
    provider: Provider | None = None,
) -> LLMResponse:
    """One call to the model, through a provider (defaults to env config).

    ``max_tokens`` is generous on purpose: Gemma is a reasoning model and spends
    tokens thinking (``reasoning_content``) before it produces visible content.
    """
    provider = provider or Provider.from_env()
    if provider.responder is not None:
        return provider.responder(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return complete_openai(
        provider,
        messages,
        model=model,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
