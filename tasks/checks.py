"""Per-chapter live acceptance checks and on-camera demos.

Each chapter appends an entry:
  ACCEPTANCE["ch-NN"] -> callable returning True/False (asserts the capability live)
  DEMOS["ch-NN"]      -> callable that prints the on-camera demonstration

A folded chapter's accept ANDs all its parts' booleans, so no proven capability is
lost. ch-00 is theory — it registers nothing.
"""

from __future__ import annotations

from collections.abc import Callable

ACCEPTANCE: dict[str, Callable[[], bool]] = {}
DEMOS: dict[str, Callable[[], None]] = {}


# ----------------------------------------------------------------------------
# ch-01 — Model only (one live call + the swappable provider seam)
# ----------------------------------------------------------------------------
def _accept_ch01_model() -> bool:
    """The real agent answers a single question via one live model call."""
    from harness import agent

    reply = agent.Agent().send("What is 2 + 2? Reply with only the number.")
    print("model replied:", repr(reply))
    return "4" in reply


def _accept_ch01_provider() -> bool:
    """The agent runs through an explicit Provider object — the swappable seam."""
    from harness import agent
    from model import Provider

    provider = Provider.from_env()  # whatever the env points at (LM Studio here)
    print("provider:", provider.base_url, provider.model)
    a = agent.Agent(provider=provider)
    reply = a.send("Reply with exactly one word: PLUGGABLE")
    print("reply:", repr(reply))
    return "pluggable" in reply.lower()


def _accept_ch01() -> bool:
    """Model only = a single live call + the swappable provider seam."""
    return _accept_ch01_model() and _accept_ch01_provider()


def _demo_ch01() -> None:
    from harness import agent
    from model import lmstudio, ollama, openrouter

    a = agent.Agent()
    print("Q: Say hello in one short sentence.")
    print("A:", a.send("Say hello in one short sentence."), "\n")
    print("— statelessness: it has no memory yet —")
    print("A1:", a.send("Your name is Gemma."))
    print("A2:", a.send("What is your name?"))
    print("  # it forgets — there's no history yet (ch-02 fixes this)\n")
    print("— same Agent, swap the provider seam (just pass provider=<one of these>) —")
    print("lmstudio  :", lmstudio().base_url)
    print("ollama    :", ollama("llama3").base_url)
    print("openrouter:", openrouter("google/gemma-3-27b-it", "sk-or-...").base_url)


ACCEPTANCE["ch-01"] = _accept_ch01
DEMOS["ch-01"] = _demo_ch01
