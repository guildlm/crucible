"""Sandbox abstraction for executing untrusted code safely.

The Docker call is hidden behind the :class:`Sandbox` interface so evaluators
depend on a small, mockable surface. In CI (no Docker) tests inject a fake
sandbox; in production :class:`DockerSandbox` runs code in a locked-down
``golang:alpine`` container with no network and tight resource limits.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

__all__ = ["SandboxResult", "Sandbox", "DockerSandbox"]


@dataclass(slots=True)
class SandboxResult:
    """Outcome of running a command set inside a sandbox.

    Attributes:
        status: One of ``ok``, ``failed``, ``timeout``, ``unavailable`` or
            ``error``. ``ok`` means every command exited zero.
        returncode: Exit code of the last command executed.
        stdout: Captured standard output (combined across commands).
        stderr: Captured standard error (combined across commands).
        timed_out: Whether execution was killed by the timeout.
        command: The final command that was run (for diagnostics).
    """

    status: str
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    command: Sequence[str] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True only when the sandbox reported a clean ``ok`` run."""
        return self.status == "ok"


class Sandbox(ABC):
    """Interface for running files + commands in an isolated environment."""

    @abstractmethod
    def available(self) -> bool:
        """Return whether the sandbox can actually execute (e.g. Docker up)."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        files: dict[str, str],
        commands: Sequence[Sequence[str]],
        *,
        timeout: float = 30.0,
    ) -> SandboxResult:
        """Write ``files`` then run ``commands`` in order inside the sandbox.

        Args:
            files: Mapping of relative path -> file contents to materialise.
            commands: Ordered list of argv command vectors. Execution stops at
                the first non-zero exit.
            timeout: Wall-clock budget (seconds) for the whole run.
        """
        raise NotImplementedError


class DockerSandbox(Sandbox):
    """Run commands inside a disposable, network-isolated Docker container.

    Security posture:
        * ``--network none`` — no outbound or inbound network.
        * ``--memory`` / ``--cpus`` / ``--pids-limit`` — resource caps.
        * ``--read-only`` root FS with a single writable bind mount for the
          workspace, mounted ``nosuid``-style via Docker defaults.
        * ``--rm`` so containers never accumulate.
        * a hard wall-clock ``timeout`` enforced by the host as a backstop.
    """

    def __init__(
        self,
        image: str = "golang:alpine",
        *,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 256,
        workdir: str = "/work",
        docker_bin: str = "docker",
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.workdir = workdir
        self.docker_bin = docker_bin

    def available(self) -> bool:
        """Return True when the Docker CLI is present and the daemon responds."""
        if shutil.which(self.docker_bin) is None:
            return False
        try:
            proc = subprocess.run(
                [self.docker_bin, "info"],
                capture_output=True,
                timeout=10,
            )
            return proc.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def _docker_argv(self, host_dir: str, command: Sequence[str]) -> list[str]:
        """Build the ``docker run`` argv for a single command."""
        return [
            self.docker_bin,
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--read-only",
            "--tmpfs",
            "/tmp:exec",
            "-e",
            "GOFLAGS=-mod=mod",
            "-e",
            "GOPROXY=off",
            "-v",
            f"{host_dir}:{self.workdir}",
            "-w",
            self.workdir,
            self.image,
            *command,
        ]

    def run(
        self,
        files: dict[str, str],
        commands: Sequence[Sequence[str]],
        *,
        timeout: float = 30.0,
    ) -> SandboxResult:
        if not self.available():
            logger.warning("Docker sandbox unavailable; failing gracefully.")
            return SandboxResult(
                status="unavailable",
                stderr="Docker is not available on this host.",
            )

        with tempfile.TemporaryDirectory(prefix="crucible-") as host_dir:
            root = Path(host_dir)
            for rel, content in files.items():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            out_parts: list[str] = []
            err_parts: list[str] = []
            last_cmd: Sequence[str] = ()
            for command in commands:
                argv = self._docker_argv(host_dir, command)
                last_cmd = command
                try:
                    proc = subprocess.run(
                        argv,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    logger.info("Sandbox command timed out: %s", command)
                    return SandboxResult(
                        status="timeout",
                        timed_out=True,
                        stdout="\n".join(out_parts),
                        stderr="\n".join(err_parts) + f"\nTimed out after {timeout}s.",
                        command=command,
                    )
                except OSError as exc:  # pragma: no cover - defensive only
                    return SandboxResult(
                        status="error", stderr=str(exc), command=command
                    )

                out_parts.append(proc.stdout)
                err_parts.append(proc.stderr)
                if proc.returncode != 0:
                    return SandboxResult(
                        status="failed",
                        returncode=proc.returncode,
                        stdout="\n".join(out_parts),
                        stderr="\n".join(err_parts),
                        command=command,
                    )

            return SandboxResult(
                status="ok",
                returncode=0,
                stdout="\n".join(out_parts),
                stderr="\n".join(err_parts),
                command=last_cmd,
            )
