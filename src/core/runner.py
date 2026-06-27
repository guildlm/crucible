"""Benchmark runner.

Loads a suite config (YAML), generates predictions via an injected model
callable, dispatches each sample to its evaluators (optionally in parallel),
and aggregates everything into a :class:`~src.core.types.BenchmarkReport`.

A suite YAML looks like::

    name: go_basic
    description: Basic Go functional checks
    evaluators:
      - go_functional
    dataset: data/go_basic.jsonl     # path relative to the suite file
    config:                          # optional per-evaluator kwargs
      llm_judge:
        offline: true

Each dataset line is a JSON object: ``{"id", "prompt", "reference",
"prediction", "metadata"}``. ``prediction`` may be omitted, in which case the
model callable is invoked with the prompt to produce it.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from src.core.registry import Evaluator, get_evaluator
from src.core.types import BenchmarkReport, EvalResult, EvalSample

logger = logging.getLogger(__name__)

__all__ = ["SuiteConfig", "BenchmarkRunner", "FakeModel", "echo_model", "render_reports"]

#: A model callable maps a prompt string to a completion string.
ModelCallable = Callable[[str], str]


@dataclass(slots=True)
class SuiteConfig:
    """Parsed representation of a suite YAML file."""

    name: str
    evaluators: list[str]
    samples: list[EvalSample]
    description: str = ""
    config: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuiteConfig":
        """Load and validate a suite config from a YAML file on disk."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "evaluators" not in raw or not raw["evaluators"]:
            raise ValueError(f"Suite {path} must declare at least one evaluator.")

        samples = cls._load_dataset(raw, path)
        return cls(
            name=str(raw.get("name", path.stem)),
            description=str(raw.get("description", "")),
            evaluators=[str(e) for e in raw["evaluators"]],
            samples=samples,
            config={k: dict(v) for k, v in (raw.get("config") or {}).items()},
        )

    @staticmethod
    def _load_dataset(raw: dict[str, Any], suite_path: Path) -> list[EvalSample]:
        """Resolve samples from an inline list or a referenced dataset file."""
        if raw.get("samples"):
            return [EvalSample.from_dict(s) for s in raw["samples"]]

        dataset = raw.get("dataset")
        if not dataset:
            raise ValueError(f"Suite {suite_path} declares neither 'samples' nor 'dataset'.")

        data_path = (suite_path.parent / dataset).resolve()
        samples: list[EvalSample] = []
        with data_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(EvalSample.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"{data_path}:{lineno}: invalid sample ({exc}).") from exc
        if not samples:
            raise ValueError(f"Dataset {data_path} contained no samples.")
        return samples


class FakeModel:
    """Deterministic model callable for tests and offline runs.

    Looks each prompt up in ``responses``; falls back to ``default``. With no
    mapping and no default it echoes the prompt, so a run is always reproducible.
    """

    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None) -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if prompt in self.responses:
            return self.responses[prompt]
        if self.default is not None:
            return self.default
        return prompt


def echo_model(prompt: str) -> str:
    """A trivial deterministic model that echoes its prompt."""
    return prompt


class BenchmarkRunner:
    """Run suites end-to-end and produce reports.

    Args:
        model: Optional prompt->completion callable used to fill in missing
            sample predictions. If ``None``, samples must already carry a
            ``prediction``.
        evaluator_config: Default kwargs per evaluator name, merged under (and
            overridden by) any config declared in the suite file.
        max_workers: Thread-pool size for evaluating samples concurrently. Use
            ``1`` for fully sequential, deterministic execution.
    """

    def __init__(
        self,
        model: ModelCallable | None = None,
        *,
        evaluator_config: dict[str, dict[str, Any]] | None = None,
        max_workers: int = 1,
    ) -> None:
        self.model = model
        self.evaluator_config = evaluator_config or {}
        self.max_workers = max(1, max_workers)

    # ----- public API -----------------------------------------------------------

    def run_suite(self, path: str | Path) -> BenchmarkReport:
        """Load a suite YAML and run it, returning a report."""
        suite = SuiteConfig.from_yaml(path)
        return self.run(suite)

    def run(self, suite: SuiteConfig) -> BenchmarkReport:
        """Execute a parsed :class:`SuiteConfig` and aggregate the results."""
        logger.info(
            "Running suite %r with %d sample(s) and evaluators %s",
            suite.name,
            len(suite.samples),
            suite.evaluators,
        )
        evaluators = self._build_evaluators(suite)
        prepared = [self._ensure_prediction(s) for s in suite.samples]

        results = self._evaluate_all(prepared, evaluators)
        report = BenchmarkReport(
            suite=suite.name,
            results=results,
            metadata={
                "description": suite.description,
                "evaluators": suite.evaluators,
                "samples": len(prepared),
            },
        )
        logger.info(
            "Suite %r complete: pass_rate=%.1f%% mean_score=%.3f",
            suite.name,
            report.pass_rate * 100,
            report.mean_score,
        )
        return report

    # ----- internals ------------------------------------------------------------

    def _build_evaluators(self, suite: SuiteConfig) -> list[Evaluator]:
        """Instantiate every evaluator named by the suite with merged config."""
        evaluators: list[Evaluator] = []
        for name in suite.evaluators:
            config = {**self.evaluator_config.get(name, {}), **suite.config.get(name, {})}
            evaluators.append(get_evaluator(name, **config))
        return evaluators

    def _ensure_prediction(self, sample: EvalSample) -> EvalSample:
        """Fill in a missing prediction by calling the model, if available."""
        if sample.prediction is not None:
            return sample
        if self.model is None:
            raise ValueError(
                f"Sample {sample.id!r} has no prediction and no model was provided."
            )
        sample.prediction = self.model(sample.prompt)
        return sample

    def _evaluate_all(
        self, samples: Sequence[EvalSample], evaluators: Sequence[Evaluator]
    ) -> list[EvalResult]:
        """Evaluate every (sample, evaluator) pair, optionally in parallel."""
        tasks = [(s, ev) for s in samples for ev in evaluators]
        if self.max_workers == 1:
            return [self._evaluate_one(s, ev) for s, ev in tasks]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda t: self._evaluate_one(*t), tasks))

    @staticmethod
    def _evaluate_one(sample: EvalSample, evaluator: Evaluator) -> EvalResult:
        """Run one evaluator on one sample, never raising on evaluator errors."""
        try:
            result = evaluator.evaluate(sample)
        except Exception as exc:  # an evaluator bug must not abort the whole run
            logger.exception("Evaluator %s failed on sample %s", evaluator.name, sample.id)
            result = EvalResult(
                score=0.0,
                passed=False,
                details={"status": "evaluator_error", "message": str(exc)},
            )
        result.evaluator = evaluator.name
        result.sample_id = sample.id
        return result


def render_reports(report: BenchmarkReport) -> tuple[str, str]:
    """Convenience helper returning ``(json_text, markdown_text)``."""
    return report.to_json(), report.to_markdown()
