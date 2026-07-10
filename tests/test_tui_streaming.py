"""TUI streaming (post-ch-14 feat).

The Textual UI renders tokens into a live agent block as they arrive, and
finalizes that same block when the turn ends — it must not leave a half-streamed
block *and* mount a duplicate final one. Driven headlessly with Textual's pilot;
the model is mocked to fire the streaming callback.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import harness.agent as agent_mod
from model import LLMResponse
from ui.tui import AgentTUI


def _streaming_chat(payload, *, on_delta=None, **kwargs):
    if on_delta:
        on_delta("reasoning", "hmm ")
        on_delta("content", "Hel")
        on_delta("content", "lo")
    return LLMResponse(content="Hello", usage={"total_tokens": 3})


def test_streaming_builds_a_live_block_then_finalizes_without_duplicating(tmp_path):
    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test() as pilot:
            with patch.object(agent_mod, "chat", side_effect=_streaming_chat):
                reply = app.agent.send("hi", on_delta=app._stream_delta)
            await pilot.pause()
            assert app._live_content == "Hello"  # content accumulated live
            assert app._live_reason_text == "hmm "  # reasoning streamed on its channel
            assert len(app.query(".msg-agent")) == 1  # exactly one streamed block

            app._turn_done(reply)
            await pilot.pause()
            assert len(app.query(".msg-agent")) == 1  # finalized in place, not duplicated
            assert app._live_body is None  # live state reset for the next turn

    asyncio.run(run())


def test_turn_done_without_streaming_still_mounts_a_block(tmp_path):
    """A turn that streamed nothing (e.g. on_delta unused) must still render its
    reply — the non-streaming path stays intact."""

    async def run():
        app = AgentTUI(sessions_dir=str(tmp_path))
        async with app.run_test() as pilot:
            app._turn_done("plain reply")
            await pilot.pause()
            assert len(app.query(".msg-agent")) == 1

    asyncio.run(run())
