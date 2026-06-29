"""Per-chapter live acceptance checks and on-camera demos.

Each chapter appends an entry:
  ACCEPTANCE["ch-NN"] -> callable returning True/False (asserts the capability live)
  DEMOS["ch-NN"]      -> callable that prints the on-camera demonstration
"""

from __future__ import annotations

from collections.abc import Callable

ACCEPTANCE: dict[str, Callable[[], bool]] = {}
DEMOS: dict[str, Callable[[], None]] = {}
