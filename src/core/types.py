"""Core data types for the Crucible evaluation framework.

These dataclasses form the contract between samples, evaluators and reports.
They are intentionally serialisable (JSON round-trips cleanly) so that
benchmark runs can be persisted, diffed and re-rendered without re-execution.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["EvalSample", "EvalResult", "BenchmarkReport"]


@dataclass(slots=True)
class EvalSample:
    """A single unit of work fed to an evaluator.

    Attributes:
        id: Stable identifier for the sample (unique within a suite).
        prompt: The instruction/input given to the model under test.
        reference: Optional gold/canonical answer used by some evaluators.
        prediction: The model completion. May be ``None`` until a model
            callable fills it in during a run.
        metadata: Arbitrary per-sample data (e.g. Go test source, rubric
            hints, banned patterns). Evaluators read what they need from here.
    """

    id: str
    prompt: str
    reference: str | None = None
    prediction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the sample."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSample":
        """Build an :class:`EvalSample` from a plain dict, ignoring extras."""
        return cls(
            id=str(data["id"]),
            prompt=data.get("prompt", ""),
            reference=data.get("reference"),
            prediction=data.get("prediction"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class EvalResult:
    """The verdict an evaluator returns for one sample.

    Attributes:
        score: A continuous quality signal in the inclusive range ``[0, 1]``.
        passed: A boolean pass/fail verdict (often ``score >= threshold``).
        details: Free-form structured diagnostics (stdout, sub-scores, etc.).
        evaluator: Name of the evaluator that produced this result. Populated
            by the runner so reports can break results down by evaluator.
        sample_id: Identifier of the originating sample. Populated by runner.
    """

    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    evaluator: str = ""
    sample_id: str = ""

    def __post_init__(self) -> None:
        # Clamp defensively: evaluators should produce [0, 1] but we never
        # want a stray value to corrupt downstream aggregate statistics.
        self.score = max(0.0, min(1.0, float(self.score)))
        self.passed = bool(self.passed)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        """Build an :class:`EvalResult` from a plain dict, ignoring extras."""
        return cls(
            score=float(data.get("score", 0.0)),
            passed=bool(data.get("passed", False)),
            details=dict(data.get("details", {})),
            evaluator=str(data.get("evaluator", "")),
            sample_id=str(data.get("sample_id", "")),
        )


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregate outcome of a benchmark run.

    Holds every per-sample :class:`EvalResult` plus convenience accessors for
    pass-rate, mean score and a per-evaluator breakdown. Renders to both JSON
    (machine-readable) and Markdown (human-readable).
    """

    suite: str
    results: list[EvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----- aggregate statistics -------------------------------------------------

    @property
    def total(self) -> int:
        """Number of evaluated results in the report."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Count of results whose ``passed`` flag is ``True``."""
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        """Fraction of results that passed, in ``[0, 1]`` (0 when empty)."""
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        """Mean of all result scores in ``[0, 1]`` (0 when empty)."""
        return statistics.fmean(r.score for r in self.results) if self.results else 0.0

    def by_evaluator(self) -> dict[str, dict[str, float | int]]:
        """Break aggregate stats down per evaluator name.

        Returns a mapping ``evaluator -> {total, passed, pass_rate, mean_score}``.
        """
        buckets: dict[str, list[EvalResult]] = defaultdict(list)
        for r in self.results:
            buckets[r.evaluator or "unknown"].append(r)

        breakdown: dict[str, dict[str, float | int]] = {}
        for name, rs in sorted(buckets.items()):
            n = len(rs)
            p = sum(1 for r in rs if r.passed)
            breakdown[name] = {
                "total": n,
                "passed": p,
                "pass_rate": p / n if n else 0.0,
                "mean_score": statistics.fmean(r.score for r in rs) if rs else 0.0,
            }
        return breakdown

    # ----- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-serialisable report including aggregates."""
        return {
            "suite": self.suite,
            "metadata": self.metadata,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "pass_rate": self.pass_rate,
                "mean_score": self.mean_score,
            },
            "by_evaluator": self.by_evaluator(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReport":
        """Reconstruct a report from :meth:`to_dict` output."""
        return cls(
            suite=str(data.get("suite", "")),
            results=[EvalResult.from_dict(r) for r in data.get("results", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkReport":
        """Reconstruct a report from a JSON string produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    # ----- rendering ------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render a readable Markdown report.

        The layout is deterministic so it can be snapshot-tested and committed
        as an artefact for review.
        """
        lines: list[str] = []
        lines.append(f"# Crucible Report: {self.suite}")
        lines.append("")
        lines.append(
            f"**Pass rate:** {self.pass_rate:.1%} "
            f"({self.passed}/{self.total}) &nbsp; "
            f"**Mean score:** {self.mean_score:.3f}"
        )
        lines.append("")

        # Per-evaluator breakdown table.
        lines.append("## By evaluator")
        lines.append("")
        lines.append("| Evaluator | Passed | Total | Pass rate | Mean score |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name, stats in self.by_evaluator().items():
            lines.append(
                f"| `{name}` | {stats['passed']} | {stats['total']} | "
                f"{stats['pass_rate']:.1%} | {stats['mean_score']:.3f} |"
            )
        lines.append("")

        # Per-sample detail table.
        lines.append("## Per-sample results")
        lines.append("")
        lines.append("| Sample | Evaluator | Result | Score | Notes |")
        lines.append("| --- | --- | :---: | ---: | --- |")
        for r in self.results:
            verdict = "✅ pass" if r.passed else "❌ fail"
            note = str(r.details.get("status", r.details.get("message", "")))
            note = note.replace("|", "\\|").replace("\n", " ")
            if len(note) > 80:
                note = note[:77] + "..."
            lines.append(
                f"| `{r.sample_id}` | `{r.evaluator}` | {verdict} | "
                f"{r.score:.3f} | {note} |"
            )
        lines.append("")
        return "\n".join(lines)
