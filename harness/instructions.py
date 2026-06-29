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

import re
from pathlib import Path


def load_agents_md(directory: str | Path = ".") -> str:
    """Return ``<directory>/AGENTS.md`` contents, or '' if it doesn't exist."""
    path = Path(directory) / "AGENTS.md"
    return path.read_text() if path.is_file() else ""


# The project's declared test command lives in a fenced block under a `## Testing`
# heading — a light convention the harness parses (no prose-guessing, no LLM call).
_TESTING_RE = re.compile(
    r"^##\s+Test(?:ing|s)?\b.*?\n```[a-zA-Z0-9]*\n\s*([^\n`]+)", re.MULTILINE | re.DOTALL
)


def test_command(directory: str | Path = ".") -> str | None:
    """The project's declared test command: the first line of the first fenced code
    block under a ``## Testing`` heading in AGENTS.md. ``None`` if there is none —
    in which case the harness has no hard gate to enforce."""
    path = Path(directory) / "AGENTS.md"
    if not path.is_file():
        return None
    m = _TESTING_RE.search(path.read_text())
    return m.group(1).strip() if m else None
