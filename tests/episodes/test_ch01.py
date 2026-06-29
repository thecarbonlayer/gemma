"""ch-01 — Model only.

Capability (persists): a single send() returns the model's reply, made through a
swappable Provider seam.

ch-01 is stateless — the agent forgets across turns. That is a *limitation*, not a
capability, so we don't pin it with a test here: ch-02 lifts it by adding history,
and its test guards the new capability. Tests that guard capabilities persist;
tests that pin a limitation retire when a later chapter removes it.

The model is patched here so the test is deterministic and offline; the real model
run lives in ``uv run accept ch-01``.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from model import LLMResponse, Provider, lmstudio, ollama, openrouter


def test_send_returns_model_content():
    with patch.object(agent_mod, "chat", return_value=LLMResponse(content="hello there")) as m:
        out = agent_mod.Agent().send("hi")
    assert out == "hello there"
    assert m.call_count == 1


# --- provider seam -----------------------------------------------------------
def test_presets_configure_endpoints():
    assert openrouter("m", "k").base_url == "https://openrouter.ai/api/v1"
    assert ollama("m").base_url == "http://localhost:11434/v1"
    assert lmstudio().model  # has a default model


def test_agent_routes_through_provider():
    seen = {}

    def fake_chat(messages, **kwargs):
        seen["provider"] = kwargs.get("provider")
        return LLMResponse(content="ok")

    p = Provider(base_url="http://example/v1", model="m", api_key="k")
    with patch.object(agent_mod, "chat", side_effect=fake_chat):
        agent_mod.Agent(provider=p).send("hi")

    assert seen["provider"] is p  # nothing above changed — only the seam
