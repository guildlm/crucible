"""Shared pytest fixtures and helpers for the Crucible test suite."""

from __future__ import annotations

from typing import Sequence

import pytest

from src.core.sandbox import Sandbox, SandboxResult


class FakeSandbox(Sandbox):
    """A scripted sandbox returning a pre-baked :class:`SandboxResult`.

    Records the files and commands it received so tests can assert on them.
    """

    def __init__(self, result: SandboxResult, *, is_available: bool = True) -> None:
        self._result = result
        self._available = is_available
        self.calls: list[tuple[dict[str, str], list[list[str]]]] = []

    def available(self) -> bool:
        return self._available

    def run(self, files, commands, *, timeout: float = 30.0) -> SandboxResult:
        self.calls.append((dict(files), [list(c) for c in commands]))
        if not self._available:
            return SandboxResult(status="unavailable", stderr="docker down")
        return self._result


@pytest.fixture
def ok_sandbox() -> FakeSandbox:
    """A sandbox that reports a clean run."""
    return FakeSandbox(SandboxResult(status="ok", returncode=0, stdout="PASS"))


@pytest.fixture
def failing_sandbox() -> FakeSandbox:
    """A sandbox that reports a failed test run."""
    return FakeSandbox(
        SandboxResult(
            status="failed",
            returncode=1,
            stdout="--- FAIL: TestAdd",
            stderr="FAIL\tsandbox",
            command=("go", "test", "./..."),
        )
    )
