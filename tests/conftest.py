"""Per-suite pytest config for the episode tests.

From ch-03 the agent auto-loads ``AGENTS.md`` from its working directory. The repo
root has one (these very instructions), so a test that constructs ``Agent()`` with
the default ``agents_dir="."`` would silently pick it up — ambient state leaking
into the assertion. The ``isolate_cwd`` fixture runs every test in a fresh empty
directory, so instruction auto-loading sees nothing unless a test stages a file
itself. This is the "something worth isolating between tests" the agent now has.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path) -> Iterator[None]:
    """Run each test in a clean temp dir so no ambient AGENTS.md is auto-loaded."""
    prev = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(prev)
