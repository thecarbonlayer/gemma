"""ch-03 — Instructions.

Capability: a system prompt (and an auto-loaded AGENTS.md) is prepended to every
call; absent by default, so earlier chapters' payloads are unchanged. Mocked for
determinism; the live check is `uv run accept ch-03`.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from harness.workspace import Workspace
from model import LLMResponse


def _capture():
    seen: list[list[dict]] = []

    def fake_chat(messages, **kwargs):
        seen.append(list(messages))
        return LLMResponse(content="ok")

    return seen, fake_chat


def test_system_message_is_prepended():
    seen, fake = _capture()
    with patch.object(agent_mod, "chat", side_effect=fake):
        agent_mod.Agent(system="You are terse.").send("hi")
    assert seen[0][0] == {"role": "system", "content": "You are terse."}


def test_no_system_by_default():
    seen, fake = _capture()
    with patch.object(agent_mod, "chat", side_effect=fake):
        agent_mod.Agent().send("hi")
    assert all(m["role"] != "system" for m in seen[0])


def test_agents_md_layers_onto_system():
    ws = Workspace()
    ws.write("AGENTS.md", "Project rule: be terse.")
    seen, fake = _capture()
    with patch.object(agent_mod, "chat", side_effect=fake):
        agent_mod.Agent(system="You are helpful.", agents_dir=str(ws.root)).send("hi")
    system_text = seen[0][0]["content"]
    assert "You are helpful." in system_text
    assert "Project rule: be terse." in system_text
