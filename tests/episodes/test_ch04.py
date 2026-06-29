"""ch-04 — Context delivery.

Capability: the harness reads any ``@path`` the user references and injects the
file contents into the call; with no reference, the payload is just the user
turn. Mocked for determinism; the live check is `uv run accept ch-04`.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from model import LLMResponse


def _capture():
    seen: list[list[dict]] = []

    def fake_chat(messages, **kwargs):
        seen.append(list(messages))
        return LLMResponse(content="ok")

    return seen, fake_chat


def test_attached_file_is_injected(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("SECRET=42")
    seen, fake = _capture()

    with patch.object(agent_mod, "chat", side_effect=fake):
        agent_mod.Agent().send(f"@{f} what is the secret?")

    payload_text = " ".join(m["content"] for m in seen[0])
    assert "SECRET=42" in payload_text


def test_no_attachment_when_no_reference():
    seen, fake = _capture()

    with patch.object(agent_mod, "chat", side_effect=fake):
        agent_mod.Agent().send("just a plain question")

    # Only the single user turn — no injected context blocks.
    assert seen[0] == [{"role": "user", "content": "just a plain question"}]
