"""The agent — the harness drive loop. Grows one primitive per chapter.

ch-01 — Model only. The whole "agent" is one stateless model call. There is no
loop, no history, no tools, no instructions. Every call is independent, which is
exactly why it can't remember anything yet (watch it forget in the demo).

The call goes through the ``model/`` seam (``chat`` + ``Provider``), so swapping
providers never touches this file — that is the seam earning its keep from day one.
"""

from __future__ import annotations

from model import Provider, chat


class Agent:
    """A thin, stateless wrapper around a single model call. No memory (yet)."""

    def __init__(self, model: str | None = None, provider: Provider | None = None) -> None:
        self.model = model
        self.provider = provider
        # NO self.messages — ch-01 is stateless; the demo's "watch it forget" depends on this.

    def send(self, user_text: str) -> str:
        """Send one message, return the model's reply. Nothing is carried over."""
        resp = chat(
            [{"role": "user", "content": user_text}],
            model=self.model,
            provider=self.provider,
        )
        return resp.content


def main() -> None:
    agent = Agent()
    print("agent ready (ch-01) — stateless: every turn is an independent model call,")
    print("so it can't remember anything yet (watch it forget). Ctrl-D to exit.")
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
