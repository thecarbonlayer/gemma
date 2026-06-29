# Build a Custom Agent Harness, from scratch

A coding agent, built in Python from the ground up, one harness primitive at a time. Each chapter
adds a single primitive, shows the capability it unlocks, and proves it against a real model. Nothing
is mocked.

> **Agent = Model + Harness + UI.** The model barely changes. The harness is the work.

## The thesis

The model is the part that almost never changes. The harness is everything around it: the loop, the
instructions, the context it sees, the tools it can call, where they run, what survives a crash, how
work is verified, and the UI that makes all of it visible. The harness is where the engineering lives,
and it is the reason two agents running the same model can behave completely differently. When an agent
"gets dumb" mid-task, it is almost always a harness problem, not a model problem.

This repo proves the point by building the harness from scratch. Each chapter adds one primitive, shows
it failing without that primitive, then shows it working with it, and runs the result against a real
model (Gemma via LM Studio locally). A chapter is not done until the agent actually does the thing.

## Who it is for

Engineers who want to understand how coding agents work by building one, not by reading a framework's
docs. The audience is comfortable with Python but new to harness engineering as a discipline. The goal
is intuition you can carry to any agent stack, not familiarity with one library.

## Setup

```bash
uv sync                       # create the env, install deps
cp .env.example .env          # point at your model endpoint (LM Studio / OpenRouter / Ollama / ...)
```

The agent talks to any OpenAI-compatible endpoint via the `model/` package (the provider seam).
Defaults target a local LM Studio server; set `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` to use
anything else.

```bash
uv run agent "your prompt"   # the REPL (from ch-01 on)
uv run tui                   # the Textual TUI (ch-14)
uv run demo ch-NN            # the demo for a chapter
```

## The chapters

The course is 15 chapters (`ch-00` … `ch-14`), each introducing one harness primitive in its mature
form.

| Ch | Title | Primitive |
|----|-------|-----------|
| 00 | What is an agent? | The frame: Model + Harness + UI (theory, no code) |
| 01 | Model only | A single model call behind a thin provider seam |
| 02 | History | Conversation state persists across turns |
| 03 | Instructions | System prompt + auto-loaded project files (AGENTS.md) |
| 04 | Context delivery | `@path` references inject file content the model can't read itself |
| 05 | Tools | The tool interface + file tools + the approval gate |
| 06 | Context management | Cache-stable assembly + compaction + door caps |
| 07 | Skills | Advertise, then load `SKILL.md` procedures on demand |
| 08 | Execution environment | Run commands in a hardened sandbox |
| 09 | Durable state + memory | JSON-L sessions on disk + episodic search |
| 10 | Orchestration | Plan steps, gate each, execute, retry |
| 11 | Subagents | Spawn isolated agents; fan out; return answers, not transcripts |
| 12 | Verification | Run candidate code against an oracle; self-verify |
| 13 | Observability | Trace every LLM/tool call as an OTel `gen_ai.*` span tree |
| 14 | UI: Textual TUI | Transcript, live trace tree, approval modal |

The build deliberately starts at `ch-00`, with the question most material skips: what separates an
agent from a chatbot or a script, and why the harness, not the model, is where the leverage is. Only
after that grounding does the build begin.

## The primitive → module map

Each primitive maps to the module that owns it.

| Primitive | Module | What it does |
|-----------|--------|--------------|
| Provider seam | `model/provider.py`, `model/openai_compatible.py`, `model/client.py` | `Provider` + a free `chat()` for any OpenAI-compatible model |
| Fake provider | `model/fake.py` | A deterministic, offline provider behind the same seam |
| Cost tracking | `model/pricing.py` | Map model ids to rates; show tokens and dollars |
| History | `harness/agent.py` | The drive loop: model, tool calls, model, with persisted messages |
| Instructions | `harness/instructions.py` | Auto-load `AGENTS.md` onto the system prompt |
| Context delivery | `harness/context.py` | Inject `@path` file content the model cannot read itself |
| Context management | `harness/compaction.py`, `harness/limits.py` | Compact the middle past a budget; clamp per-item sizes at the door |
| Tools | `harness/tools.py`, `harness/workspace.py` | A tool registry; file read/write/edit over a scoped workspace |
| Skills | `harness/skills.py` | Load `SKILL.md` procedures; progressive disclosure |
| Execution environment | `harness/sandbox.py` | Run commands in hardened Docker or a scrubbed local subprocess |
| Durable state + memory | `harness/memory.py` | JSON-L sessions on disk; keyword search across past sessions |
| Orchestration | `harness/orchestrator.py` | Plan steps, gate each, execute, retry |
| Subagents | `harness/subagents.py` | Spawn isolated agents; fan out; return answers, not transcripts |
| Verification | `harness/verification.py` | Run candidate code against an assertion; return proof, not trust |
| Observability | `harness/observability.py`, `harness/events.py` | A flat trace plus an OpenTelemetry `gen_ai.*` span tree, with an exporter seam |
| UI | `ui/tui.py` | A Textual TUI: transcript, live trace tree, approval modal |

## How it is built (and verified)

Two gates, because "the tests pass" and "the agent actually works" are different claims.

- **`uv run verify`** is the floor: ruff (format + lint), mypy, pytest, smoke import. Deterministic,
  offline.
- **`uv run accept ch-NN`** is the truth: the real agent against a real model, asserting the chapter's
  capability end to end. A chapter is not done until the agent can really do the thing on a real model.

Each chapter is its own commit, tagged `ch-00` … `ch-14`. Check one out to see the project as it stood
at that point in the build:

```bash
git checkout ch-05     # the project at chapter 5 (Tools)
git checkout main      # back to the latest
```

## Layout

```
model/          # the provider seam + costing: provider / openai_compatible / fake / client / pricing
harness/        # the loop + every primitive: agent, context, tools, memory, skills, sandbox,
                #   orchestrator, subagents, verification, observability, events, ...
ui/             # the Textual TUI (the only package that imports textual/rich)
tasks/          # uv-run tooling: verify / accept / demo / tui
tests/episodes/ # one behavioral test file per chapter (test_ch01..test_ch14)
```

The code is three packages with dependencies pointing one way: `ui/` → `harness/` → `model/`. The core
never imports the UI. `model/` talks to a model, `harness/` is the loop and every primitive, `ui/`
renders it. The agent loop lives only in `harness/agent.py`; the REPL is the `agent` console script.
