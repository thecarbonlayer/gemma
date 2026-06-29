"""Execution environment (ch-08) — the harness runs code, the model never does.

The model only ever asks; the harness executes, inside a boundary. The sandbox
prefers hardened Docker (``--network none``, non-root, scoped workdir) and falls
back to a scoped local subprocess when no Docker daemon is available.

"Start closed": no network, a fresh isolated workdir, and a scrubbed environment
(no inherited credentials), so untrusted code never sees the host's secrets. The
sandbox is the backstop, not the only defense.

The seam was introduced minimal at ch-05 (one chokepoint for code execution);
this is the hardening — the boundary that makes that chokepoint trustworthy.
Give it a ``workdir`` and the command runs in that persistent directory, so a
bash command can see a file a write tool just created (the workspace seam).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

from harness.tools import Tool

# Minimal environment handed to sandboxed commands — note the absence of secrets.
_SCRUBBED_ENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C"}


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    backend: str


class Sandbox:
    def __init__(
        self,
        image: str = "busybox",
        timeout: float = 15.0,
        prefer_docker: bool = True,
        trusted: bool = False,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.prefer_docker = prefer_docker
        # trusted: run in the REAL environment (uv/PATH/deps visible), unscrubbed.
        # For a coding agent working on your own project running your own test
        # command — the approval gate is the control, not network-none isolation.
        self.trusted = trusted
        self._docker: bool | None = None

    def _docker_up(self) -> bool:
        if not self.prefer_docker:
            return False
        if self._docker is None:
            try:
                self._docker = (
                    subprocess.run(
                        ["docker", "info"],
                        capture_output=True,
                        timeout=5,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.SubprocessError):
                self._docker = False
        return self._docker

    def run(self, command: str, workdir: str | None = None) -> SandboxResult:
        # A workdir makes the sandbox operate on a persistent workspace (bind-mounted
        # in docker, cwd locally) instead of a throwaway dir.
        if self.trusted:
            return self._run_local(command, workdir)
        if self._docker_up():
            return self._run_docker(command, workdir)
        return self._run_local(command, workdir)

    def _run_docker(self, command: str, workdir: str | None) -> SandboxResult:
        # Hardened: no network, non-root, capabilities dropped, writable only in /work.
        # /work is a throwaway tmpfs unless a workspace is bind-mounted.
        work = ["-v", f"{workdir}:/work"] if workdir else ["--tmpfs", "/work:rw,size=16m"]
        argv = [
            "docker", "run", "--rm",
            "--network", "none",
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--memory", "256m",
            "--pids-limit", "128",
            "--read-only",
            *work,
            "-w", "/work",
            self.image,
            "sh", "-c", command,
        ]  # fmt: skip
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        return SandboxResult(proc.stdout, proc.stderr, proc.returncode, "docker")

    def _run_local(self, command: str, workdir: str | None) -> SandboxResult:
        # Fallback: scrubbed env + timeout. Uses the persistent workspace if given,
        # else a fresh throwaway dir. (network is NOT isolated here — that needs Docker.)
        cwd = workdir or tempfile.mkdtemp(prefix="sandbox-")
        # trusted → the real environment (your test runner needs uv/PATH/deps);
        # otherwise the scrubbed env (untrusted code sees no host secrets).
        env = os.environ.copy() if self.trusted else dict(_SCRUBBED_ENV, HOME=cwd, TMPDIR=cwd)
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return SandboxResult(
            proc.stdout, proc.stderr, proc.returncode, "trusted" if self.trusted else "local"
        )


def bash_tool(sandbox: Sandbox, workdir: str | None = None) -> Tool:
    """A bash tool whose commands run inside the sandbox. With a workdir, commands
    run in the persistent workspace (so they see files the edit tools wrote)."""

    def run_bash(command: str) -> str:
        r = sandbox.run(command, workdir=workdir)
        body = (r.stdout + r.stderr).strip()
        return f"[exit {r.exit_code} via {r.backend}]\n{body}"

    return Tool(
        name="bash",
        description="Run a shell command in an isolated sandbox and return its output.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        func=run_bash,
    )
