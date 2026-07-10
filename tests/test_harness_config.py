"""The editable surface has a shape, not a content (harness_config).

These tests pin the *structure* of the config primitive — it loads, it's frozen,
the loader rejects malformed files loudly, and the legacy module-level names are
pure re-exports of the config values. They deliberately do NOT pin any knob's
*value*: the whole point of the surface is that values change (an editor bumps
the version and rewrites the file); the verifier must not entangle with them.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from harness.harness_config import CONFIG, CONFIG_PATH, HarnessConfig, load_config


def _valid_raw() -> dict:
    """The checked-in config as a plain dict — the base for mutation tests."""
    return json.loads(CONFIG_PATH.read_text())


def _write(tmp_path: Path, raw: dict) -> Path:
    p = tmp_path / "harness_config.json"
    p.write_text(json.dumps(raw))
    return p


def test_config_loads_and_is_typed():
    assert isinstance(CONFIG, HarnessConfig)
    assert isinstance(CONFIG.version, int)
    assert isinstance(CONFIG.system_prompt, str)
    assert isinstance(CONFIG.max_tool_steps, int)
    assert isinstance(CONFIG.default_context_limit, int)
    assert isinstance(CONFIG.verify_attempts, int)
    assert isinstance(CONFIG.require_run, bool)
    assert isinstance(CONFIG.max_item_chars, int)
    assert isinstance(CONFIG.compaction_prompt, str)
    assert isinstance(CONFIG.memory_search_limit, int)
    assert isinstance(CONFIG.attach_pattern, str)


def test_set_fields_are_frozensets_of_str():
    for value in (CONFIG.approval_tools, CONFIG.code_extensions):
        assert isinstance(value, frozenset)
        assert all(isinstance(x, str) for x in value)


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.max_tool_steps = 99  # type: ignore[misc]


def test_load_config_roundtrips_the_checked_in_file():
    assert load_config(CONFIG_PATH) == CONFIG


def test_unknown_key_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["mystery_knob"] = 7
    with pytest.raises(ValueError, match="unknown"):
        load_config(_write(tmp_path, raw))


def test_missing_field_is_rejected(tmp_path):
    raw = _valid_raw()
    del raw["max_tool_steps"]
    with pytest.raises(ValueError, match="missing"):
        load_config(_write(tmp_path, raw))


def test_wrong_type_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["max_tool_steps"] = "six"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, raw))


def test_bool_is_not_an_int(tmp_path):
    # bool is a subclass of int in Python; the door must not let True through
    # where an integer knob is expected.
    raw = _valid_raw()
    raw["version"] = True
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, raw))


def test_non_positive_int_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["max_tool_steps"] = 0
    with pytest.raises(ValueError, match="positive"):
        load_config(_write(tmp_path, raw))


def test_non_compiling_attach_pattern_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["attach_pattern"] = "@(unclosed"
    with pytest.raises(ValueError, match="attach_pattern"):
        load_config(_write(tmp_path, raw))


def test_groupless_attach_pattern_is_rejected(tmp_path):
    # The use site extracts the path via group(1); a pattern with no capture
    # group compiles fine but would break every @path delivery.
    raw = _valid_raw()
    raw["attach_pattern"] = "@\\S+"
    with pytest.raises(ValueError, match="attach_pattern"):
        load_config(_write(tmp_path, raw))


def test_non_string_in_set_field_is_rejected(tmp_path):
    raw = _valid_raw()
    raw["approval_tools"] = ["bash", 3]
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, raw))


def test_non_object_document_is_rejected(tmp_path):
    p = tmp_path / "harness_config.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_config(p)


def test_reexports_equal_config_values():
    """Legacy names stay importable, but are pure views of the config."""
    import harness.agent as agent
    import harness.compaction as compaction
    import harness.context as context
    import harness.limits as limits

    assert agent.DEFAULT_SYSTEM == CONFIG.system_prompt
    assert agent.MAX_TOOL_STEPS == CONFIG.max_tool_steps
    assert agent.DEFAULT_CONTEXT_LIMIT == CONFIG.default_context_limit
    assert agent.APPROVAL_TOOLS == CONFIG.approval_tools
    assert agent.CODE_EXTENSIONS == CONFIG.code_extensions
    assert limits.MAX_ITEM_CHARS == CONFIG.max_item_chars
    assert compaction.COMPACTION_PROMPT == CONFIG.compaction_prompt
    assert context._ATTACH.pattern == CONFIG.attach_pattern


def test_ctor_defaults_come_from_config():
    from harness.agent import Agent

    a = Agent(agents_dir=str(Path(__file__).parent))  # dodge the ambient AGENTS.md
    assert a.context_limit == CONFIG.default_context_limit
    assert a.verify_attempts == CONFIG.verify_attempts
    assert a.require_run == CONFIG.require_run


def test_tui_approval_tools_read_the_config():
    import ui.tui as tui

    assert tui.APPROVAL_TOOLS == CONFIG.approval_tools
