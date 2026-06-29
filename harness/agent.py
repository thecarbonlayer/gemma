"""The agent — the harness drive loop. Grows one primitive per chapter.

ch-04 — Context delivery. The model can only see what's in the prompt; it can't
open a file on its own. So when the user writes ``@notes.txt``, the *harness*
reads that file and injects its contents as context before the real turn.

``send`` now runs a pre-loop: ``deliver`` scans the user text for ``@path``
references and returns a block per readable file; each block is appended as its
own context message, and only then does the actual user turn go in. The model
answers from contents the harness fetched for it.

The instruction layers from ch-03 are unchanged: ``_system_text`` joins the
built-in ``system`` prompt and any auto-loaded ``AGENTS.md``; ``_payload`` puts
that at the head of the history. The single ``chat`` call still goes through the
``model/`` seam, so swapping providers never touches this file.
"""

from __future__ import annotations

from harness.context import deliver
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
        """Inject any @path files, append the turn, replay, keep the reply."""
        for block in deliver(user_text):  # @file references → injected context
            self.messages.append({"role": "user", "content": f"Context file:\n{block}"})
        self.messages.append({"role": "user", "content": user_text})
        resp = chat(self._payload(), model=self.model, provider=self.provider)
        self.messages.append({"role": "assistant", "content": resp.content})
        return resp.content


def main() -> None:
    agent = Agent(system=DEFAULT_SYSTEM)
    print("agent ready (ch-04) — reference files with @path. Ctrl-D to exit.")
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
