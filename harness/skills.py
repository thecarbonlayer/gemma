"""Skills (ch-07).

A skill is a *directory* containing a ``SKILL.md`` file with YAML frontmatter
(``name`` + ``description``) followed by instructions — the agentskills.io
format. Only the name and description are advertised in the prompt; the model
loads the full body on demand with the read_file tool (progressive disclosure).
Skills are not tools — a tool is a capability ("run pytest"), a skill is a
procedure ("how we cut a release").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Read top-level ``key: value`` pairs from the leading ``---`` YAML block."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        # only top-level keys (skip nested/indented lines like a metadata map)
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


def load_skills(directory: str | Path) -> list[Skill]:
    """Load skills from ``<directory>/<name>/SKILL.md`` (agentskills.io layout)."""
    root = Path(directory)
    if not root.is_dir():
        return []
    skills = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        meta = _parse_frontmatter(skill_md.read_text())
        skills.append(
            Skill(
                name=meta.get("name", skill_md.parent.name),
                description=meta.get("description", ""),
                path=skill_md,
            )
        )
    return skills


def skills_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = [
        "You have skills available. When one applies, use the read_file tool to "
        "read its file, then follow it exactly:"
    ]
    lines += [f"- {s.name}: {s.description} (file: {s.path})" for s in skills]
    return "\n".join(lines)
