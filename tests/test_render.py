"""Print / non-interactive renderers (post-ch-14 feat).

One-shot ``agent "prompt" --format {plain,json,transcript}`` needs three text
shapes. These are pure functions over the agent's messages + tracer, with no
``textual`` and no ``ui`` import, so print mode runs without a TUI and the
one-way deps (``ui`` → ``harness`` → ``model``) stay intact.
"""

from __future__ import annotations

import json

from harness.observability import Tracer
from harness.render import render_json, render_plain, render_transcript


def test_render_plain_is_the_reply_text():
    assert render_plain("the answer is 42") == "the answer is 42"


def test_render_json_carries_reply_and_trace_totals():
    tr = Tracer(model="openai/gpt-4o-mini")
    tr.turn_start()
    tr.record_llm(
        {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}, 0.2
    )
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    out = render_json("yo", tr, messages)
    obj = json.loads(out)  # must be valid JSON

    assert obj["reply"] == "yo"
    assert obj["messages"] == 2
    assert obj["totals"]["tokens"] == 1_000_000
    assert obj["totals"]["cost"] == 0.15  # priced from usage
    assert obj["totals"]["llm_calls"] == 1


def test_render_json_without_a_tracer_still_valid():
    obj = json.loads(render_json("done", None, []))
    assert obj["reply"] == "done"
    assert obj["totals"] == {}


def test_render_transcript_shows_every_message_and_the_trace():
    tr = Tracer(model="google/gemma-4-26b-a4b")
    tr.turn_start()
    tr.record_tool("bash", 0.01, args="ls", result="file.py", status="ok")
    messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash"}}]},
        {"role": "tool", "content": "file.py"},
        {"role": "assistant", "content": "there is file.py"},
    ]

    out = render_transcript(messages, tr)

    assert "user" in out and "list files" in out
    assert "assistant" in out and "there is file.py" in out
    assert "bash" in out  # the tool step is visible
    assert "tool" in out  # the trace timeline rendered
