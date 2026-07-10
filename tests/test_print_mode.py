"""Non-interactive print mode (post-ch-14 feat).

``agent "prompt" --format {plain,json,transcript}`` runs exactly one turn and
emits the chosen shape. Non-interactive can't prompt at a TTY, so it is
fail-closed: approval-required tools are denied unless ``-y/--yes`` is given.
Driven offline through the fake provider.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.agent import _approver, run_once
from model import fake


def test_approver_is_fail_closed_by_default():
    # No approver → the agent's gate denies bash/write/edit (its fail-closed path).
    assert _approver(False) is None


def test_approver_yes_auto_approves_gated_tools():
    approve = _approver(True)
    assert approve is not None
    assert approve("bash", '{"command": "ls"}') is True


def _run(fmt, tmp_path):
    return run_once(
        "what is 6 times 7?",
        provider=fake(scripted=lambda msgs: "42"),
        fmt=fmt,
        session="print",
        sessions_dir=str(tmp_path),
        workspace_root=str(tmp_path),
        agents_dir=str(tmp_path),  # empty dir → no AGENTS.md, no test gate
    )


def test_run_once_plain_returns_the_reply(tmp_path):
    assert _run("plain", tmp_path) == "42"


def test_run_once_json_is_machine_readable(tmp_path):
    obj = json.loads(_run("json", tmp_path))
    assert obj["reply"] == "42"
    assert "totals" in obj


def test_run_once_transcript_shows_prompt_and_reply(tmp_path):
    out = _run("transcript", tmp_path)
    assert "what is 6 times 7?" in out
    assert "42" in out


def test_print_mode_is_stateless_by_default(tmp_path):
    """Each one-shot invocation is independent: no shared session accumulates across
    calls, and nothing is persisted to disk (default session is ephemeral)."""

    def once():
        return run_once(
            "say hi",
            provider=fake(scripted=lambda msgs: "hi"),
            fmt="json",
            sessions_dir=str(tmp_path),
            workspace_root=str(tmp_path),
            agents_dir=str(tmp_path),
        )

    first = json.loads(once())
    second = json.loads(once())
    assert first["messages"] == second["messages"]  # no growth across calls
    assert not list(Path(tmp_path).glob("*.jsonl"))  # stateless — nothing persisted
