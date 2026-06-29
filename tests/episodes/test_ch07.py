"""ch-07 — Skills.

Capability: skills are advertised (name + description from SKILL.md frontmatter)
in the system prompt; the full body is loaded on demand via read_file (tested
live in accept ch-07). Skills follow the agentskills.io layout: a directory per
skill containing SKILL.md.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from harness.skills import load_skills, skills_prompt
from model import LLMResponse


def _write_skill(root, name, description, body="body here"):
    d = root / name
    d.mkdir()
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")


def test_load_and_prompt(tmp_path):
    _write_skill(tmp_path, "foo", "does foo")
    skills = load_skills(tmp_path)
    assert skills[0].name == "foo"
    assert "does foo" in skills[0].description
    prompt = skills_prompt(skills)
    assert "read_file" in prompt and "foo" in prompt
    # the body is NOT advertised — only the description
    assert "body here" not in prompt


def test_agent_advertises_skills_in_system(tmp_path):
    _write_skill(tmp_path, "foo", "does foo")
    skills = load_skills(tmp_path)
    seen: list[list[dict]] = []

    def fake_chat(messages, **kwargs):
        seen.append(list(messages))
        return LLMResponse(content="ok")

    with patch.object(agent_mod, "chat", side_effect=fake_chat):
        agent_mod.Agent(system="base", skills=skills).send("hi")

    sys_msg = seen[0][0]
    assert sys_msg["role"] == "system"
    assert "base" in sys_msg["content"] and "foo" in sys_msg["content"]
