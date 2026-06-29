"""On-camera demos: run a chapter's demonstration against a real model.

uv run demo ch-02
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, os.getcwd())
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: uv run demo ch-NN")
        return 2
    chapter = argv[0]

    from tasks.checks import DEMOS

    demo = DEMOS.get(chapter)
    if demo is None:
        print(f"no demo registered for '{chapter}'")
        return 2
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
