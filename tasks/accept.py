"""Live acceptance gate: run the real agent against a real model and assert the
chapter's actual capability. REQUIRED before a code commit.

  uv run accept ch-02
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, os.getcwd())
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: uv run accept ch-NN")
        return 2
    chapter = argv[0]

    from tasks.checks import ACCEPTANCE

    check = ACCEPTANCE.get(chapter)
    if check is None:
        print(f"no live acceptance check registered for '{chapter}'")
        return 2

    model = os.environ.get("LLM_MODEL", "google/gemma-4-26b-a4b")
    base = os.environ.get("LLM_BASE_URL", "http://192.168.189.144:1234/v1")
    print(f"== live acceptance: {chapter}  (model={model} @ {base}) ==", flush=True)
    try:
        ok = bool(check())
    except Exception as exc:  # noqa: BLE001
        print(f"ACCEPT ERROR: {exc}")
        return 1
    print("\nACCEPT OK" if ok else "\nACCEPT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
