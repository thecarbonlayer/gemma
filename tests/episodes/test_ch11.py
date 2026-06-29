"""ch-11 — Subagents.

Capability: independent subtasks run in isolated subagents and fan out, with
results returned in order.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from harness.subagents import delegate_tool, fan_out, run_subagent
from model import LLMResponse


def _echo_chat(messages, **kwargs):
    # Echo the last user message so we can see which task each subagent ran.
    last = next((m for m in reversed(messages) if m["role"] == "user"), {"content": ""})
    return LLMResponse(content="reply:" + last["content"])


def test_run_subagent_isolated():
    with patch.object(agent_mod, "chat", side_effect=_echo_chat):
        assert run_subagent("task A") == "reply:task A"


def test_fan_out_preserves_order():
    with patch.object(agent_mod, "chat", side_effect=_echo_chat):
        out = fan_out(["a", "b", "c"])
    assert out == ["reply:a", "reply:b", "reply:c"]


def test_delegate_tool():
    tool = delegate_tool()
    assert tool.name == "delegate"
    with patch.object(agent_mod, "chat", side_effect=_echo_chat):
        assert tool.func(task="sub") == "reply:sub"
