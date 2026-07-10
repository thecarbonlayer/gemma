"""Print / non-interactive renderers (post-ch-14 feat).

One-shot ``agent "prompt" --format {plain,json,transcript}`` emits its result in a
chosen shape for scripting and CI. These are pure formatting functions over the
agent's messages and its ``Tracer`` — no ``textual``, no ``ui`` import — so print
mode runs without a TUI and the one-way deps (``ui`` → ``harness`` → ``model``)
stay intact.

- ``plain``      — just the final answer (what a shell pipeline wants).
- ``json``       — a machine-readable object: reply + trace totals (tokens, cost).
- ``transcript`` — every message and tool step, then the trace timeline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.observability import Tracer


def render_plain(reply: str) -> str:
    """The final answer text, as-is."""
    return reply


def render_json(reply: str, tracer: Tracer | None, messages: list[dict]) -> str:
    """A JSON object: the reply, the message count, and the run's trace totals
    (tokens, cost, llm/tool calls, seconds). No tracer → empty totals."""
    payload = {
        "reply": reply,
        "messages": len(messages),
        "totals": tracer.totals() if tracer is not None else {},
    }
    return json.dumps(payload, indent=2)


def render_transcript(messages: list[dict], tracer: Tracer | None) -> str:
    """Every message and tool step as readable text, then the trace timeline."""
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", "") or "")
        calls = m.get("tool_calls") or []
        if calls:
            names = ", ".join(tc.get("function", {}).get("name", "?") for tc in calls)
            content = f"{content}[tool_calls: {names}]" if content else f"[tool_calls: {names}]"
        lines.append(f"{role}: {content}")
    if tracer is not None and tracer.events:
        lines.append("")
        lines.append("--- trace ---")
        lines.append(tracer.timeline())
    return "\n".join(lines)
