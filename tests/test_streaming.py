"""Streaming deltas (post-ch-14 feat).

The model seam can stream: fire a callback per token as they arrive, and still
return the *same* ``LLMResponse`` the blocking path returns — so nothing above the
seam (tracer, tool parsing, compaction, the verification gate) changes.

The HTTP/SSE plumbing is a thin wrapper; the assembly logic is a pure function
over already-parsed stream chunks, tested here offline with canned chunks that
mirror what an OpenAI-compatible endpoint (LM Studio / OpenRouter) emits.
"""

from __future__ import annotations

from harness.agent import Agent
from model import fake
from model.openai_compatible import assemble_stream


def _collector() -> tuple[list[tuple[str, str]], object]:
    """A two-arg ``OnDelta`` sink plus the list it appends ``(channel, text)`` to."""
    seen: list[tuple[str, str]] = []
    return seen, lambda channel, text: seen.append((channel, text))


def _content_chunk(piece: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}


def _reasoning_chunk(piece: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"reasoning_content": piece}, "finish_reason": None}]}


def test_content_deltas_concatenate_and_fire_callback():
    seen, on_delta = _collector()
    chunks = [_content_chunk("Hel"), _content_chunk("lo"), _content_chunk("!")]
    resp = assemble_stream(chunks, on_delta)
    assert resp.content == "Hello!"
    assert seen == [("content", "Hel"), ("content", "lo"), ("content", "!")]


def test_reasoning_deltas_stream_on_their_own_channel():
    seen, on_delta = _collector()
    chunks = [_reasoning_chunk("think"), _reasoning_chunk("ing"), _content_chunk("done")]
    resp = assemble_stream(chunks, on_delta)
    assert resp.reasoning == "thinking"
    assert resp.content == "done"
    assert seen == [("reasoning", "think"), ("reasoning", "ing"), ("content", "done")]


def _tool_frag(frag: dict) -> dict:
    return {"choices": [{"index": 0, "delta": {"tool_calls": [frag]}}]}


def test_tool_calls_reassemble_by_index():
    chunks = [
        _tool_frag(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator", "arguments": ""},
            }
        ),
        _tool_frag({"index": 0, "function": {"arguments": '{"expr'}}),
        _tool_frag({"index": 0, "function": {"arguments": 'ession": "1+1"}'}}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    resp = assemble_stream(chunks, None)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "calculator"
    assert tc["function"]["arguments"] == '{"expression": "1+1"}'


def test_usage_and_finish_reason_captured_from_final_chunks():
    chunks = [
        _content_chunk("hi"),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
    ]
    resp = assemble_stream(chunks, None)
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def test_no_callback_is_allowed():
    resp = assemble_stream([_content_chunk("ok")], None)
    assert resp.content == "ok"


def test_fake_provider_replays_content_through_on_delta():
    """A ``responder`` provider (fake) has no network to stream, but the seam still
    exercises the streaming path: ``chat`` replays the final content through the
    callback so offline runs stream deterministically."""
    seen, on_delta = _collector()
    a = Agent(provider=fake(scripted=lambda msgs: "PONG"))
    reply = a.send("ping", on_delta=on_delta)
    assert reply == "PONG"
    assert "".join(text for ch, text in seen if ch == "content") == "PONG"
