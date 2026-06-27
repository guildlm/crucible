"""Go functional evaluator.

Compiles and tests model-generated Go code inside a sandbox. The actual code
execution is delegated to a :class:`~src.core.sandbox.Sandbox`, so the Docker
dependency is fully mockable and the evaluator degrades gracefully when no
sandbox is available.

Sample contract:
    * ``sample.prediction`` — the Go source under test (package ``sandbox``).
    * ``sample.metadata["tests"]`` — Go test source (``*_test.go`` content).
    * ``sample.metadata["module"]`` — optional module name (default ``sandbox``).
"""

from __future__ import annotations

import logging

from src.core.registry import Evaluator, register
from src.core.sandbox import DockerSandbox, Sandbox
from src.core.types import EvalResult, EvalSample

logger = logging.getLogger(__name__)

__all__ = ["GoFunctionalEvaluator"]


@register("go_functional")
class GoFunctionalEvaluator(Evaluator):
    """Evaluate Go code by vetting, building and testing it in a sandbox.

    The pipeline runs ``go vet`` then ``go test`` (which implies a build). A
    sample passes only when both succeed. ``go vet`` failures are surfaced as a
    distinct status so static-analysis problems are visible in reports.
    """

    name = "go_functional"

    def __init__(
        self,
        sandbox: Sandbox | None = None,
        *,
        module: str = "sandbox",
        timeout: float = 30.0,
        run_vet: bool = True,
    ) -> None:
        """Args:
        sandbox: Execution backend. Defaults to a :class:`DockerSandbox`.
        module: Go module name used in the generated ``go.mod``.
        timeout: Per-run wall-clock budget passed to the sandbox.
        run_vet: Whether to run ``go vet`` before the test stage.
        """
        self.sandbox = sandbox if sandbox is not None else DockerSandbox()
        self.module = module
        self.timeout = timeout
        self.run_vet = run_vet

    def evaluate(self, sample: EvalSample) -> EvalResult:
        code = sample.prediction or ""
        tests = str(sample.metadata.get("tests", ""))
        module = str(sample.metadata.get("module", self.module))

        if not code.strip():
            return EvalResult(
                score=0.0,
                passed=False,
                details={"status": "empty_prediction", "message": "No Go code to evaluate."},
            )

        files = {
            "go.mod": f"module {module}\n\ngo 1.21\n",
            "main.go": code,
        }
        if tests.strip():
            files["main_test.go"] = tests

        commands: list[list[str]] = []
        if self.run_vet:
            commands.append(["go", "vet", "./..."])
        if tests.strip():
            commands.append(["go", "test", "./...", "-run", ".", "-count=1"])
        else:
            # No tests provided: a successful build is the strongest signal.
            commands.append(["go", "build", "./..."])

        logger.info("Running Go sandbox for sample %s", sample.id)
        result = self.sandbox.run(files, commands, timeout=self.timeout)

        details = {
            "status": result.status,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": list(result.command),
        }

        if result.status == "unavailable":
            return EvalResult(
                score=0.0,
                passed=False,
                details={**details, "message": "Sandbox unavailable (Docker required)."},
            )
        if result.status == "timeout":
            return EvalResult(
                score=0.0,
                passed=False,
                details={**details, "message": f"Execution timed out after {self.timeout}s."},
            )
        if result.ok:
            return EvalResult(score=1.0, passed=True, details=details)

        # Non-zero exit: distinguish vet vs build/test failures for reporting.
        failing = " ".join(result.command)
        message = "go vet failed" if "vet" in failing else "build or tests failed"
        return EvalResult(score=0.0, passed=False, details={**details, "message": message})
