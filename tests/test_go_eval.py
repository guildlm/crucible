"""Tests for the Go functional evaluator using a mocked sandbox."""

from __future__ import annotations

from src.core.sandbox import SandboxResult
from src.evaluators.go_eval import GoFunctionalEvaluator
from src.core.types import EvalSample
from tests.conftest import FakeSandbox

_CODE = "package sandbox\n\nfunc Add(a, b int) int { return a + b }\n"
_TESTS = (
    "package sandbox\n\nimport \"testing\"\n\n"
    "func TestAdd(t *testing.T) { if Add(2,3) != 5 { t.Fatal(\"bad\") } }\n"
)


def _sample() -> EvalSample:
    return EvalSample(
        id="add",
        prompt="write Add",
        prediction=_CODE,
        metadata={"tests": _TESTS, "module": "sandbox"},
    )


def test_pass_path(ok_sandbox: FakeSandbox):
    ev = GoFunctionalEvaluator(sandbox=ok_sandbox)
    result = ev.evaluate(_sample())
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["status"] == "ok"

    # Sandbox received the expected files and commands.
    files, commands = ok_sandbox.calls[0]
    assert "main.go" in files and "main_test.go" in files and "go.mod" in files
    assert ["go", "vet", "./..."] in commands
    assert any(c[:2] == ["go", "test"] for c in commands)


def test_fail_path(failing_sandbox: FakeSandbox):
    ev = GoFunctionalEvaluator(sandbox=failing_sandbox)
    result = ev.evaluate(_sample())
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["status"] == "failed"


def test_sandbox_unavailable_fails_gracefully():
    sandbox = FakeSandbox(SandboxResult(status="ok"), is_available=False)
    ev = GoFunctionalEvaluator(sandbox=sandbox)
    result = ev.evaluate(_sample())
    assert result.passed is False
    assert result.details["status"] == "unavailable"
    assert "Sandbox unavailable" in result.details["message"]


def test_timeout_reported():
    sandbox = FakeSandbox(SandboxResult(status="timeout", timed_out=True))
    ev = GoFunctionalEvaluator(sandbox=sandbox, timeout=5.0)
    result = ev.evaluate(_sample())
    assert result.passed is False
    assert result.details["status"] == "timeout"


def test_empty_prediction():
    sandbox = FakeSandbox(SandboxResult(status="ok"))
    ev = GoFunctionalEvaluator(sandbox=sandbox)
    result = ev.evaluate(EvalSample(id="e", prompt="p", prediction="   "))
    assert result.passed is False
    assert result.details["status"] == "empty_prediction"
    assert sandbox.calls == []  # never reached the sandbox


def test_vet_can_be_disabled(ok_sandbox: FakeSandbox):
    ev = GoFunctionalEvaluator(sandbox=ok_sandbox, run_vet=False)
    ev.evaluate(_sample())
    _, commands = ok_sandbox.calls[0]
    assert ["go", "vet", "./..."] not in commands
