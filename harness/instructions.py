"""Project instructions — AGENTS.md.

The built-in system prompt sets the agent's baseline behavior. But per-project
rules ("you are Gemma", "always run the tests", a house style) shouldn't have to
be typed every turn. So the harness auto-loads ``AGENTS.md`` from the working
directory and layers it onto the system prompt — always on. Same convention as
Codex, Claude Code, and pi.

It's a *layer* on top of the built-in system prompt, not a replacement. No file →
empty string → nothing changes.
"""

from __future__ import annotations

from pathlib import Path


def load_agents_md(directory: str | Path = ".") -> str:
    """Return ``<directory>/AGENTS.md`` contents, or '' if it doesn't exist."""
    path = Path(directory) / "AGENTS.md"
    return path.read_text() if path.is_file() else ""
