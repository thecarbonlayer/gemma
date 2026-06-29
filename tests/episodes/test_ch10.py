"""ch-10 — Orchestration.

Capability: plan a task into steps, execute them in order with an approval gate,
and fall back gracefully when the plan isn't valid JSON.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from harness import orchestrator as orch_mod
from harness.orchestrator import Orchestrator
from model import LLMResponse


def _planner_aware(plan_json: str, step_reply: str = "done"):
    def fake_chat(messages, **kwargs):
        first = messages[0].get("content", "") if messages else ""
        if "planner" in first.lower():
            return LLMResponse(content=plan_json)
        return LLMResponse(content=step_reply)

    return fake_chat


def test_plans_and_executes_in_order():
    fake = _planner_aware('["step one", "step two"]')
    with (
        patch.object(orch_mod, "chat", side_effect=fake),
        patch.object(agent_mod, "chat", side_effect=fake),
    ):
        res = Orchestrator().run("do a thing")
    assert res.plan == ["step one", "step two"]
    assert len(res.results) == 2 and res.final == "done"


def test_plan_falls_back_on_bad_json():
    with patch.object(orch_mod, "chat", return_value=LLMResponse(content="not json at all")):
        assert Orchestrator()._plan("task X") == ["task X"]


def test_approval_gate_skips_steps():
    fake = _planner_aware('["a", "b"]', step_reply="ran")
    with (
        patch.object(orch_mod, "chat", side_effect=fake),
        patch.object(agent_mod, "chat", side_effect=fake),
    ):
        res = Orchestrator().run("t", approve=lambda step: step != "b")
    assert "[skipped] b" in res.results
