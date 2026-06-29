"""ch-09 — Durable state + memory.

Capability: a session is persisted to disk and a fresh agent resumes it. Folded
in: episodic retrieval — keyword search across stored sessions returns matching
chunks, and the search_memory tool surfaces facts from sessions not in the
current context.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from harness.memory import (
    load_session,
    save_session,
    search_memory_tool,
    search_sessions,
)
from model import LLMResponse


def test_session_round_trip(tmp_path):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    save_session("s1", msgs, base=tmp_path)
    assert load_session("s1", base=tmp_path) == msgs
    assert load_session("missing", base=tmp_path) == []


def test_agent_resumes_from_disk(tmp_path):
    with patch.object(agent_mod, "chat", return_value=LLMResponse(content="ok")):
        a = agent_mod.Agent(session="demo", sessions_dir=str(tmp_path))
        a.send("remember: the one with the mark is Teja")

    # A brand-new agent (simulating a restart) reloads the prior conversation.
    b = agent_mod.Agent(session="demo", sessions_dir=str(tmp_path))
    assert b.messages  # resumed, not empty
    assert any("Teja" in m["content"] for m in b.messages)


# --- episodic retrieval / text search ----------------------------------------
def _seed(base):
    save_session(
        "past-a",
        [
            {"role": "user", "content": "Note: the warehouse passcode is GOGO-77."},
            {"role": "assistant", "content": "Noted."},
        ],
        base=base,
    )
    save_session(
        "past-b",
        [{"role": "user", "content": "Raveena is Karishma."}],
        base=base,
    )


def test_search_finds_the_right_chunk(tmp_path):
    _seed(tmp_path)
    hits = search_sessions("warehouse passcode", base=tmp_path)
    assert hits
    assert "GOGO-77" in hits[0]["content"]
    assert hits[0]["session"] == "past-a"


def test_search_empty_query_and_no_match(tmp_path):
    _seed(tmp_path)
    assert search_sessions("", base=tmp_path) == []
    assert search_sessions("nonexistent zzz", base=tmp_path) == []


def test_search_memory_tool(tmp_path):
    _seed(tmp_path)
    tool = search_memory_tool(base=tmp_path)
    assert tool.name == "search_memory"
    out = tool.func(query="passcode")
    assert "GOGO-77" in out


def test_tool_registers_and_runs_via_registry(tmp_path):
    _seed(tmp_path)
    from harness.tools import default_tools

    reg = default_tools()
    reg.register(search_memory_tool(base=tmp_path))
    assert "GOGO-77" in reg.call("search_memory", '{"query": "passcode"}')
    # sanity: the agent module still imports/constructs fine
    assert agent_mod.Agent(tools=reg)
