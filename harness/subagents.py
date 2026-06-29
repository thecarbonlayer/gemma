"""Subagents (ch-11).

Split work into bounded loops, each a fresh agent with its own isolated context
and tools. A subagent returns the answer, not its transcript, so the main
window stays clean. Independent subtasks fan out in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from harness.tools import Tool, ToolRegistry, default_tools

DEFAULT_WORKER_SYSTEM = "You are a focused worker. Do exactly the subtask and answer concisely."


def run_subagent(
    task: str,
    *,
    system: str | None = None,
    model: str | None = None,
    tools: ToolRegistry | None = None,
) -> str:
    from harness.agent import Agent  # lazy: avoids an import cycle at module load

    sub = Agent(system=system or DEFAULT_WORKER_SYSTEM, tools=tools or default_tools(), model=model)
    return sub.send(task)


def fan_out(tasks: list[str], *, model: str | None = None, max_workers: int = 4) -> list[str]:
    """Run subtasks in parallel, each in its own isolated subagent. Order preserved."""
    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        return list(pool.map(lambda t: run_subagent(t, model=model), tasks))


def delegate_tool(model: str | None = None) -> Tool:
    """A tool that lets a main agent delegate a self-contained subtask to a subagent."""

    def delegate(task: str) -> str:
        return run_subagent(task, model=model)

    return Tool(
        name="delegate",
        description="Delegate a self-contained subtask to a fresh subagent and get its result.",
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
        func=delegate,
    )


def fan_out_tool(model: str | None = None) -> Tool:
    """A tool that lets the model split work into independent subtasks and run them
    in parallel, each in its own isolated subagent. Results come back labeled and
    ordered, so the model can read them as one block."""

    def fan_out_call(tasks: list[str]) -> str:
        results = fan_out(tasks, model=model)
        return "\n\n".join(
            f"[subtask {i}] {task}\n{result}"
            for i, (task, result) in enumerate(zip(tasks, results, strict=False), 1)
        )

    return Tool(
        name="fan_out",
        description=(
            "Run several independent subtasks in parallel, each in its own fresh "
            "subagent, and get back their labeled results. Use for work that splits "
            "cleanly into pieces that don't depend on each other."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The independent subtasks to run in parallel.",
                }
            },
            "required": ["tasks"],
        },
        func=fan_out_call,
    )
