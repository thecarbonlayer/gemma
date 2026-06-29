"""Durable state (ch-09) + episodic retrieval (ch-16).

Conversation is not state. The harness persists the session as JSON-L (one
message per line, easy to append) so a killed agent can resume by reloading it.
The session boundary is the kill point — nothing survives unless it's written.

ch-16: a log isn't *memory* until you can recover the right slice. ``search_sessions``
does keyword text search across all stored sessions (no embeddings); ``search_memory_tool``
lets the model pull matching chunks from sessions that aren't in the current context.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.tools import Tool

DEFAULT_DIR = ".sessions"


def _path(session_id: str, base: str | Path = DEFAULT_DIR) -> Path:
    return Path(base) / f"{session_id}.jsonl"


def save_session(session_id: str, messages: list[dict], base: str | Path = DEFAULT_DIR) -> None:
    path = _path(session_id, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")


def load_session(session_id: str, base: str | Path = DEFAULT_DIR) -> list[dict]:
    path = _path(session_id, base)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_path(session_id: str, base: str | Path = DEFAULT_DIR) -> Path:
    # A subdir so trace files are not picked up by the *.jsonl session globs.
    return Path(base) / "traces" / f"{session_id}.jsonl"


def save_trace(session_id: str, rows: list[dict], base: str | Path = DEFAULT_DIR) -> None:
    """Persist a session's trace events (ch-24) next to its messages, so the trace
    pane survives a restart instead of resetting to empty."""
    path = _trace_path(session_id, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_trace(session_id: str, base: str | Path = DEFAULT_DIR) -> list[dict]:
    path = _trace_path(session_id, base)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def delete_session(session_id: str, base: str | Path = DEFAULT_DIR) -> None:
    """Wipe a session's persisted messages and trace (ch-24 ``/reset``). Idempotent —
    missing files are fine, since reset should work whether or not anything was saved."""
    for path in (_path(session_id, base), _trace_path(session_id, base)):
        path.unlink(missing_ok=True)


def list_sessions(base: str | Path = DEFAULT_DIR) -> list[dict]:
    """List persisted sessions for the UI (ch-24): name, message count, mtime.

    Sorted most-recently-modified first so the active/recent ones surface at the
    top of the sessions pane. Reuses the same JSON-L files as load/save."""
    base_dir = Path(base)
    if not base_dir.is_dir():
        return []
    out: list[dict] = []
    for path in base_dir.glob("*.jsonl"):
        try:
            messages = sum(1 for line in path.read_text().splitlines() if line.strip())
        except OSError:
            continue
        out.append({"name": path.stem, "messages": messages, "mtime": path.stat().st_mtime})
    out.sort(key=lambda s: s["mtime"], reverse=True)
    return out


def search_sessions(query: str, base: str | Path = DEFAULT_DIR, limit: int = 5) -> list[dict]:
    """Keyword text search across all stored sessions. Returns the best-matching
    messages as {session, role, content}, ranked by how many query terms appear."""
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []
    base_dir = Path(base)
    if not base_dir.is_dir():
        return []
    scored: list[tuple[int, dict]] = []
    for path in sorted(base_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            content = str(msg.get("content", "") or "").lower()
            score = sum(term in content for term in terms)
            if score:
                scored.append(
                    (
                        score,
                        {
                            "session": path.stem,
                            "role": msg.get("role"),
                            "content": msg.get("content"),
                        },
                    )
                )
    scored.sort(key=lambda s: s[0], reverse=True)
    return [m for _, m in scored[:limit]]


def search_memory_tool(base: str | Path = DEFAULT_DIR) -> Tool:
    """A tool the model calls to recall facts from earlier sessions."""

    def search_memory(query: str) -> str:
        hits = search_sessions(query, base=base)
        if not hits:
            return "no matching memory found"
        return "\n".join(f"[{h['session']}] {h['role']}: {h['content']}" for h in hits)

    return Tool(
        name="search_memory",
        description="Search past sessions for relevant facts by keyword.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        func=search_memory,
    )
