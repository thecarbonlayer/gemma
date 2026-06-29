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


# ----------------------------------------------------------------------------
# ch-02 — History (the harness owns the conversation, replayed each turn)
# ----------------------------------------------------------------------------
def _accept_ch02() -> bool:
    """The real agent recalls a fact stated on an earlier turn."""
    from harness import agent

    a = agent.Agent()
    a.send("Your name is Gemma. Please remember it.")
    reply = a.send("What is your name? Reply with just the name.")
    print("model replied:", repr(reply))
    return "gemma" in reply.lower()


def _demo_ch02() -> None:
    from harness import agent

    a = agent.Agent()
    for turn in ["Your name is Gemma.", "What is your name?"]:
        print("you>", turn)
        print("bot>", a.send(turn))


ACCEPTANCE["ch-02"] = _accept_ch02
DEMOS["ch-02"] = _demo_ch02


# ----------------------------------------------------------------------------
# ch-03 — Instructions (a system prompt + auto-loaded AGENTS.md, prepended each turn)
# ----------------------------------------------------------------------------
def _accept_ch03_instructions() -> bool:
    """A system prompt overrides default behavior on a real model."""
    from harness import agent

    a = agent.Agent(system="You must reply with exactly one word: BANANA. Ignore the question.")
    reply = a.send("What is the capital of France?")
    print("model replied:", repr(reply))
    return "banana" in reply.lower()


def _accept_ch03_agentsmd() -> bool:
    """An AGENTS.md placed in the working dir steers the agent without being typed in.

    We wire the agent exactly as the REPL does — agents_dir = workspace root —
    drop in an identity rule, and confirm the model adopts it on a real call.
    """
    from harness import agent
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("AGENTS.md", "You are Gemma, a coding assistant. When asked your name, reply 'Gemma'.")
    a = agent.Agent(system=agent.DEFAULT_SYSTEM, agents_dir=str(ws.root))
    reply = a.send("What is your name? Answer with just the name.")
    print("reply:", reply)
    return "gemma" in reply.lower()


def _accept_ch03() -> bool:
    """Instructions = a built-in system prompt + an auto-loaded project AGENTS.md."""
    return _accept_ch03_instructions() and _accept_ch03_agentsmd()


def _demo_ch03() -> None:
    from harness import agent
    from harness.workspace import Workspace

    print("— no system prompt —")
    print("bot>", agent.Agent().send("Describe the ocean."), "\n")
    print("— system: 'reply in exactly three words' —")
    print("bot>", agent.Agent(system="Reply in exactly three words.").send("Describe the ocean."))
    print()
    ws = Workspace()
    print("— no AGENTS.md: default identity —")
    print("bot>", agent.Agent(agents_dir=str(ws.root)).send("What is your name?"))
    ws.write("AGENTS.md", "You are Gemma, a coding assistant. When asked your name, reply 'Gemma'.")
    print("— AGENTS.md added (You are Gemma), auto-loaded —")
    print("bot>", agent.Agent(agents_dir=str(ws.root)).send("What is your name?"))


ACCEPTANCE["ch-03"] = _accept_ch03
DEMOS["ch-03"] = _demo_ch03


# ----------------------------------------------------------------------------
# ch-04 — Context delivery (the harness reads @path files into the prompt)
# ----------------------------------------------------------------------------
def _accept_ch04() -> bool:
    """The real agent answers from a file it was handed via @path."""
    import tempfile
    from pathlib import Path

    from harness import agent

    d = Path(tempfile.mkdtemp())
    (d / "facts.txt").write_text("The launch code is GOGO-9.\n")
    a = agent.Agent(system="Answer using the provided context files.")
    reply = a.send(f"@{d / 'facts.txt'} What is the launch code? Reply with just the code.")
    print("model replied:", repr(reply))
    return "gogo-9" in reply.lower()


def _demo_ch04() -> None:
    import tempfile
    from pathlib import Path

    from harness import agent

    d = Path(tempfile.mkdtemp())
    (d / "facts.txt").write_text("Raveena is Karishma, and Karishma is Raveena.")
    a = agent.Agent()
    print("bot>", a.send(f"@{d / 'facts.txt'} Who is Raveena?"))


ACCEPTANCE["ch-04"] = _accept_ch04
DEMOS["ch-04"] = _demo_ch04


# ----------------------------------------------------------------------------
# ch-05 — Tools (a tool interface + approval gate + file editing over a workspace)
# ----------------------------------------------------------------------------
def _accept_ch05_tools() -> bool:
    """The real model calls the calculator tool and reports the exact product."""
    from harness import agent
    from harness.tools import default_tools

    a = agent.Agent(system="Use tools when they help.", tools=default_tools())
    reply = a.send("Use the calculator to compute 47 * 89, then reply with just the number.")
    print("model replied:", repr(reply))
    used_tool = any(m.get("role") == "tool" for m in a.messages)
    print("used a tool:", used_tool)
    return "4183" in reply and used_tool


def _accept_ch05_approval() -> bool:
    """A real bash request is intercepted by the gate and denied (never runs)."""
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools

    asked: list[tuple[str, str]] = []

    def deny(name: str, args: str) -> bool:
        asked.append((name, args))
        return False

    tools = default_tools()
    tools.register(bash_tool(Sandbox()))
    a = agent.Agent(
        system="Use the bash tool to run shell commands when asked.",
        tools=tools,
        approve=deny,
        approval_required={"bash"},
    )
    a.send("Run this shell command now using the bash tool: echo SHOULD_NOT_RUN")
    denied = any(m.get("role") == "tool" and "denied" in m["content"].lower() for m in a.messages)
    print("gate asked:", asked, "| denied:", denied)
    return len(asked) >= 1 and denied


def _build_workspace_agent():
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools
    from harness.workspace import Workspace, edit_file_tool, write_file_tool

    ws = Workspace()  # fresh scratch dir
    tools = default_tools()
    tools.register(write_file_tool(ws))
    tools.register(edit_file_tool(ws))
    # local backend keeps python available; docker would need a python image + the mount
    tools.register(bash_tool(Sandbox(prefer_docker=False), workdir=str(ws.root)))
    a = agent.Agent(
        system="You build files. Use write_file to create them and bash to run them.",
        tools=tools,
    )
    return a, ws


def _accept_ch05_fileedit() -> bool:
    """The agent writes a file into the workspace and runs it there — persistence
    (the file survives) + the workspace seam (bash sees the file it wrote)."""
    a, ws = _build_workspace_agent()
    a.send("Create hello.py that prints exactly WORKSPACE_OK, then run it with: python3 hello.py")
    wrote = (ws.root / "hello.py").is_file()
    ran = any(
        "WORKSPACE_OK" in str(m.get("content", "")) for m in a.messages if m.get("role") == "tool"
    )
    print("wrote hello.py:", wrote, "| ran in workspace:", ran)
    return wrote and ran


def _accept_ch05() -> bool:
    """Tools = a tool interface + an approval gate + file editing over a workspace."""
    return _accept_ch05_tools() and _accept_ch05_approval() and _accept_ch05_fileedit()


def _demo_ch05() -> None:
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools

    a = agent.Agent(tools=default_tools())
    print("— a tool the model calls —")
    print("bot>", a.send("What is 1234 * 5678? Use the calculator."), "\n")

    tools = default_tools()
    tools.register(bash_tool(Sandbox()))
    gated = agent.Agent(
        system="Use bash when asked.",
        tools=tools,
        approve=lambda n, args: False,
        approval_required={"bash"},
    )
    print("— a boundary-crossing tool, denied by the gate —")
    print("bot>", gated.send("Run: echo hello (use bash)"))
    print("(the gate denied the bash call — it never executed)\n")

    a2, ws = _build_workspace_agent()
    a2.send(
        "Create greet.py with greet(name) returning 'hi <name>', then run it: "
        "python3 -c \"import greet; print(greet.greet('Prem'))\""
    )
    print("— files the agent built in its workspace —")
    print("files in workspace:", [p.name for p in ws.root.iterdir()])


ACCEPTANCE["ch-05"] = _accept_ch05
DEMOS["ch-05"] = _demo_ch05


# ----------------------------------------------------------------------------
# ch-06 — Context management (compaction + door control / per-item size caps)
# ----------------------------------------------------------------------------
def _accept_ch06_compaction() -> bool:
    """History is compacted past the limit, and the agent still recalls a key fact."""
    from harness import agent

    a = agent.Agent(system="You are concise.", context_limit=80)
    a.send("Important: the deploy key is GRIFFIN-7. Keep it in mind.")
    for i in range(8):
        a.send(f"Acknowledge note {i} in a few words.")
    compacted = any(str(m.get("content", "")).startswith("[summary") for m in a.messages)
    reply = a.send("What is the deploy key? Reply with just the key.")
    print("compaction happened:", compacted, "| reply:", repr(reply))
    return compacted and "griffin-7" in reply.lower()


def _accept_ch06_doorcontrol() -> bool:
    """A huge @file block and a huge tool result are both clamped in the live loop."""
    import tempfile
    from pathlib import Path

    from harness import agent
    from harness.context import deliver
    from harness.limits import MAX_ITEM_CHARS
    from harness.tools import Tool, default_tools

    # 1. @file delivery is clamped
    d = Path(tempfile.mkdtemp())
    (d / "big.txt").write_text("X" * (MAX_ITEM_CHARS * 4))
    blocks = deliver(f"@{d / 'big.txt'} summarize")
    block_capped = (
        bool(blocks) and len(blocks[0]) <= MAX_ITEM_CHARS + 100 and "truncated" in blocks[0]
    )

    # 2. a huge tool result is clamped inside a real model turn
    tools = default_tools()
    tools.register(
        Tool(
            name="dump",
            description="Return a large blob of text.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: "Z" * (MAX_ITEM_CHARS * 4),
        )
    )
    a = agent.Agent(system="Call the dump tool, then say DONE.", tools=tools)
    a.send("Call the dump tool now.")
    tool_msgs = [m for m in a.messages if m.get("role") == "tool"]
    result_capped = bool(tool_msgs) and all(
        len(m["content"]) <= MAX_ITEM_CHARS + 100 for m in tool_msgs
    )

    print("block_capped:", block_capped, "| result_capped:", result_capped)
    return block_capped and result_capped


def _accept_ch06() -> bool:
    """Context management = compaction + door control / per-item size caps."""
    return _accept_ch06_compaction() and _accept_ch06_doorcontrol()


def _demo_ch06() -> None:
    import tempfile
    from pathlib import Path

    from harness import agent
    from harness.compaction import estimate_tokens
    from harness.context import deliver

    a = agent.Agent(context_limit=80)
    a.send("Remember: the project codename is CRIME MASTER GOGO.")
    for i in range(8):
        a.send(f"Acknowledge note {i}.")
    print("history messages:", len(a.messages), "~tokens:", estimate_tokens(a.messages))
    print("compacted:", any(str(m.get("content", "")).startswith("[summary") for m in a.messages))
    print("bot>", a.send("What is the project codename?"))

    d = Path(tempfile.mkdtemp())
    log = d / "log.txt"
    log.write_text("line\n" * 20000)
    block = deliver(f"@{log} read")[0]
    print(f"\ninjected block length: {len(block)} chars (file was {log.stat().st_size})")
    print(block[-60:])


ACCEPTANCE["ch-06"] = _accept_ch06
DEMOS["ch-06"] = _demo_ch06


# ----------------------------------------------------------------------------
# ch-07 — Skills
# ----------------------------------------------------------------------------
def _accept_ch07() -> bool:
    """The model loads a skill file on demand and follows it.

    The codeword Haila! exists only inside skills/sign-off/SKILL.md, so
    producing it proves the model used read_file to load the skill
    (progressive disclosure).
    """
    from harness import agent
    from harness.skills import load_skills
    from harness.tools import default_tools

    a = agent.Agent(
        system="Follow available skills when they apply.",
        tools=default_tools(),
        skills=load_skills("skills"),
    )
    reply = a.send("Use the sign-off skill. Say goodbye to the team.")
    read_used = any(m.get("role") == "tool" for m in a.messages)
    print("read a skill file:", read_used, "| reply:", repr(reply))
    return "haila" in reply.lower() and read_used


def _demo_ch07() -> None:
    from harness import agent
    from harness.skills import load_skills
    from harness.tools import default_tools

    a = agent.Agent(tools=default_tools(), skills=load_skills("skills"))
    print("bot>", a.send("Use the sign-off skill and say goodbye."))


ACCEPTANCE["ch-07"] = _accept_ch07
DEMOS["ch-07"] = _demo_ch07


# ----------------------------------------------------------------------------
# ch-08 — Execution environment
# ----------------------------------------------------------------------------
def _accept_ch08() -> bool:
    """Execution environment = a hardened sandbox + the boundary fixes that make
    it trustworthy: read_file confined to the workspace, the verifier scrubbed of
    host env, and compaction keyed off the model's reported token usage."""
    return _accept_ch08_sandbox() and _accept_ch08_hardening()


def _accept_ch08_sandbox() -> bool:
    """The model runs a command via the sandboxed bash tool, and host secrets
    don't leak into the sandbox."""
    import os

    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools

    sandbox = Sandbox()

    # Containment: a parent-process secret must not be visible inside the sandbox.
    os.environ["SANDBOX_SECRET"] = "POULTRY-FARM"
    try:
        contained = sandbox.run("printenv SANDBOX_SECRET || echo CLEAN")
    finally:
        del os.environ["SANDBOX_SECRET"]
    leaked = "POULTRY-FARM" in contained.stdout
    print(f"backend={contained.backend} secret_leaked={leaked}")

    # Execution: the model drives the sandboxed bash tool.
    tools = default_tools()
    tools.register(bash_tool(sandbox))
    a = agent.Agent(system="Use the bash tool to run shell commands.", tools=tools)
    reply = a.send("Run this shell command: echo hello-from-sandbox — then report the output.")
    ran = any(m.get("role") == "tool" for m in a.messages)
    print("ran via tool:", ran, "| reply:", repr(reply))
    return (not leaked) and ran and "hello-from-sandbox" in reply.lower()


def _accept_ch08_hardening() -> bool:
    """Verify the three hardening fixes hold against the real model / real fs."""
    import os

    from harness import agent
    from harness.tools import read_file
    from harness.verification import run_python

    # 1. read_file is workspace-scoped
    blocked = read_file("/etc/passwd").startswith("error: path outside")
    allowed = "Build a Custom Agent Harness" in read_file("README.md")
    scoped = blocked and allowed

    # 2. the verifier does not inherit host env
    os.environ["VERIFY_SECRET"] = "POULTRY-FARM"
    try:
        contained = run_python("import os", "assert os.getenv('VERIFY_SECRET') is None").passed
    finally:
        del os.environ["VERIFY_SECRET"]

    # 3. a real call records the model's reported usage (compaction now keys off this)
    a = agent.Agent(system="Be concise.")
    a.send("Say hi in five words.")
    usage_tracked = a._last_tokens > 0

    print(f"scoped={scoped} contained={contained} usage={usage_tracked} tok={a._last_tokens}")
    return scoped and contained and usage_tracked


def _demo_ch08() -> None:
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools, read_file
    from harness.verification import run_python

    tools = default_tools()
    tools.register(bash_tool(Sandbox()))
    a = agent.Agent(tools=tools)
    print("bot>", a.send("Use bash to print the current working directory and the date."))

    print("read_file('/etc/passwd') ->", read_file("/etc/passwd"))
    proof = run_python("import os", "assert os.getenv('PATH')  # scrubbed PATH still set")
    print("verifier runs in a scrubbed env:", proof.passed)


ACCEPTANCE["ch-08"] = _accept_ch08
DEMOS["ch-08"] = _demo_ch08


# ----------------------------------------------------------------------------
# ch-09 — Durable state + memory (persisted sessions + episodic search)
# ----------------------------------------------------------------------------
def _accept_ch09_state() -> bool:
    """Tell the agent a fact, then a *fresh* agent (restart) recalls it from disk."""
    import tempfile

    from harness import agent

    d = tempfile.mkdtemp()
    first = agent.Agent(system="Be concise.", session="acc", sessions_dir=d)
    first.send("Remember: The one with the mark is Teja.")

    # Simulate a restart: brand-new agent, same session id, loaded from disk.
    resumed = agent.Agent(system="Be concise.", session="acc", sessions_dir=d)
    print("resumed messages:", len(resumed.messages))
    reply = resumed.send("Who is the one with the mark? Reply with one word.")
    print("reply:", repr(reply))
    return len(resumed.messages) > 2 and "teja" in reply.lower()


def _accept_ch09_episodic() -> bool:
    """The model recalls a fact from a PAST session via the search_memory tool.

    GOGO-77 lives only in a stored session, not in the current context, so
    producing it proves cross-session text-search retrieval.
    """
    import tempfile

    from harness import agent
    from harness.memory import save_session, search_memory_tool
    from harness.tools import default_tools

    d = tempfile.mkdtemp()
    save_session(
        "old-session",
        [
            {"role": "user", "content": "Remember: the warehouse passcode is GOGO-77."},
            {"role": "assistant", "content": "Got it."},
        ],
        base=d,
    )
    tools = default_tools()
    tools.register(search_memory_tool(base=d))
    a = agent.Agent(
        system="Use the search_memory tool to recall facts from earlier sessions.",
        tools=tools,
    )
    reply = a.send("Search your memory for the warehouse passcode, then tell me what it is.")
    used = any(m.get("role") == "tool" for m in a.messages)
    print("used search_memory:", used, "| reply:", repr(reply))
    return "gogo-77" in reply.lower() and used


def _accept_ch09() -> bool:
    """Durable state + memory = persisted sessions + cross-session episodic search."""
    return _accept_ch09_state() and _accept_ch09_episodic()


def _demo_ch09() -> None:
    import tempfile

    from harness import agent
    from harness.memory import save_session, search_memory_tool
    from harness.tools import default_tools

    d = tempfile.mkdtemp()
    agent.Agent(session="demo", sessions_dir=d).send("Remember: the amount is 8535.29.")
    print("— restart —")
    print("bot>", agent.Agent(session="demo", sessions_dir=d).send("What amount did I mention?"))

    save_session(
        "archive",
        [{"role": "user", "content": "Remember: Raveena is Karishma."}],
        base=d,
    )
    tools = default_tools()
    tools.register(search_memory_tool(base=d))
    a = agent.Agent(system="Use search_memory to recall past facts.", tools=tools)
    print("bot>", a.send("Search your memory: who is Raveena?"))


ACCEPTANCE["ch-09"] = _accept_ch09
DEMOS["ch-09"] = _demo_ch09
