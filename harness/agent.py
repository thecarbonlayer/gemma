"""The agent — the harness drive loop. Grows one primitive per chapter.

ch-03 — Instructions. ch-02 gave the agent memory of the *conversation*. But its
behavior was set per turn: nothing told it how to act across turns. A system
prompt fixes that — set the behavior once and the harness prepends it to every
call. Persistent behavior without repeating yourself.

The harness assembles two instruction layers into that prompt:

  - ``system``     — the built-in/base behavior (passed in, or DEFAULT_SYSTEM).
  - ``AGENTS.md``  — per-project rules auto-loaded from the working directory.

``_system_text`` joins the non-empty layers; ``_payload`` puts the result at the
head of the history. The single ``chat`` call still goes through the ``model/``
seam, so swapping providers never touches this file.
"""

from __future__ import annotations

from harness.instructions import load_agents_md
from model import Provider, chat

DEFAULT_SYSTEM = "You are a concise, helpful coding assistant."


class Agent:
    """A model wrapped in conversation memory and a persistent system prompt."""

    def __init__(
        self,
        model: str | None = None,
        provider: Provider | None = None,
        system: str | None = None,
        agents_dir: str = ".",
    ) -> None:
        self.model = model
        self.provider = provider
        self.system = system
        self.agents_dir = agents_dir  # where AGENTS.md is auto-loaded from
        self.messages: list[dict] = []

    def _system_text(self) -> str:
        """The instruction layer = built-in system prompt + project AGENTS.md."""
        parts = [p for p in (self.system, load_agents_md(self.agents_dir)) if p]
        return "\n\n".join(parts)

    def _payload(self) -> list[dict]:
        """System prompt first (if any), then the full conversation history."""
        sys_text = self._system_text()
        head = [{"role": "system", "content": sys_text}] if sys_text else []
        return head + self.messages

    def send(self, user_text: str) -> str:
        """Append the turn, replay history behind the system prompt, keep the reply."""
        self.messages.append({"role": "user", "content": user_text})
        resp = chat(self._payload(), model=self.model, provider=self.provider)
        self.messages.append({"role": "assistant", "content": resp.content})
        return resp.content


def main() -> None:
    agent = Agent(system=DEFAULT_SYSTEM)
    print("agent ready (ch-03) — with a system prompt. Ctrl-D to exit.")
    while True:
        try:
            user = input("you> ")
        except EOFError:
            print()
            break
        if not user.strip():
            continue
        print("bot>", agent.send(user))


if __name__ == "__main__":
    main()
