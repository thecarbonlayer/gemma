"""Per-suite pytest config for the episode tests.

Nothing to set up yet at ch-01: the agent is a single stateless call with no
working directory, no ambient files, no state to isolate. This file grows a real
fixture once a later chapter gives the agent something worth isolating between
tests (e.g. auto-loaded instructions in ch-03).

It exists from ch-01 so the ``tests/`` package has one obvious home for test
infrastructure as the suite accretes, mirroring how ``harness/`` accretes the
agent's primitives.
"""

from __future__ import annotations
