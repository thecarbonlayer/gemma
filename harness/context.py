"""Context delivery (ch-04).

The model can't read files; the harness does. ``deliver`` scans user text for
``@path`` references, reads those files, and returns them as context blocks the
agent injects into the prompt — turning "look at @notes.txt" into the file's
actual contents in the window.

The blocks here are raw and uncapped. A huge file would flood the window; that's
a real problem, and door control (clamping each block) is the job of a later
chapter. Right now the point is just: the harness, not the model, opens the file.
"""

from __future__ import annotations

import re
from pathlib import Path

_ATTACH = re.compile(r"@(\S+)")


def deliver(user_text: str) -> list[str]:
    """Return a context block for each readable ``@path`` referenced in the text."""
    blocks: list[str] = []
    for match in _ATTACH.finditer(user_text):
        path = Path(match.group(1))
        if path.is_file():
            try:
                body = path.read_text()
            except OSError:
                continue
            blocks.append(f"--- {path} ---\n{body}")
    return blocks
