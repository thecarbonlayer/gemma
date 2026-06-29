"""The workspace — a directory the agent owns.

A small, path-safe wrapper around a directory: write, read, and edit files whose
contents survive across calls. Every path is confined to the workspace root, so a
bad write can't escape to the host. By default it's a fresh scratch dir, so an
experiment can't touch your real project; point ``root`` at a real repo only if
you mean it.

At ch-03 this is just the class — the accept check stages an ``AGENTS.md`` here to
prove auto-loaded instructions work. The write/edit *tools* that let the model
build a multi-file project arrive at ch-05, once the agent has a tool interface.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = (
            Path(root).resolve() if root else Path(tempfile.mkdtemp(prefix="workspace-")).resolve()
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return p

    def write(self, path: str, content: str) -> str:
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {path} ({len(content)} chars)"

    def read(self, path: str) -> str:
        p = self._safe(path)
        return p.read_text() if p.is_file() else f"error: no such file: {path}"

    def edit(self, path: str, old: str, new: str) -> str:
        p = self._safe(path)
        if not p.is_file():
            return f"error: no such file: {path}"
        text = p.read_text()
        if old not in text:
            return f"error: text to replace not found in {path}"
        p.write_text(text.replace(old, new, 1))
        return f"edited {path}"
