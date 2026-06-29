"""A fake provider — the second implementation of the model seam.

The provider seam earns its keep the moment it has more than one implementation.
``fake`` is that second one: same ``Provider`` shape, but it answers from a script
instead of the network, so ``verify`` is deterministic and fully offline.

Usage::

    Agent(provider=fake(scripted=lambda msgs: "PONG"))   # echo a fixed reply
    Agent(provider=fake(scripted=["a", "b"]))             # reply per turn, then last
    Agent(provider=fake())                                # always "ok"
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import count

from model.provider import LLMResponse, Provider


class FakeProvider:
    """A deterministic responder. ``scripted`` is either a callable
    ``(messages) -> str`` or a list of replies (consumed in order, then the last
    reply repeats); when absent, every call returns ``default``."""

    def __init__(
        self,
        *,
        scripted: Callable[[list[dict]], str] | list[str] | None = None,
        default: str = "ok",
    ) -> None:
        self.scripted = scripted
        self.default = default
        self._calls = count()

    def __call__(self, messages: list[dict], **_kwargs) -> LLMResponse:
        i = next(self._calls)
        if callable(self.scripted):
            content = self.scripted(messages)
        elif isinstance(self.scripted, list) and self.scripted:
            content = self.scripted[min(i, len(self.scripted) - 1)]
        else:
            content = self.default
        return LLMResponse(content=content, finish_reason="stop")


def fake(
    *,
    scripted: Callable[[list[dict]], str] | list[str] | None = None,
    default: str = "ok",
) -> Provider:
    """Build a ``Provider`` backed by a deterministic, offline responder."""
    return Provider(
        base_url="fake://local",
        model="fake",
        api_key="x",
        responder=FakeProvider(scripted=scripted, default=default),
    )
