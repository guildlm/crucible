"""Tests for the BenchmarkRunner aggregation and rendering."""

from __future__ import annotations

import pytest

from src.core.registry import Evaluator, register
from src.core.runner import BenchmarkRunner, FakeModel, SuiteConfig
from src.core.types import EvalResult, EvalSample


@register("len_threshold")
class LenThreshold(Evaluator):
    """Pass when the prediction is at least ``min_len`` characters."""

    name = "len_threshold"

    def __init__(self, *, min_len: int = 3) -> None:
        self.min_len = min_len

    def evaluate(self, sample: EvalSample) -> EvalResult:
        n = len(sample.prediction or "")
        passed = n >= self.min_len
        return EvalResult(score=1.0 if passed else 0.0, passed=passed, details={"len": n})


def _suite(**overrides) -> SuiteConfig:
    samples = [
        EvalSample(id="s1", prompt="aa"),       # echo -> "aa" (len 2) -> fail
        EvalSample(id="s2", prompt="hello"),    # echo -> "hello" (len 5) -> pass
    ]
    base = dict(
        name="unit",
        evaluators=["len_threshold"],
        samples=samples,
        config={"len_threshold": {"min_len": 3}},
    )
    base.update(overrides)
    return SuiteConfig(**base)


def test_runner_fills_predictions_and_aggregates():
    runner = BenchmarkRunner(model=FakeModel())  # echoes prompt
    report = runner.run(_suite())

    assert report.total == 2
    assert report.passed == 1
    assert report.pass_rate == 0.5
    assert 0.0 <= report.mean_score <= 1.0
    # Each result carries provenance.
    assert {r.sample_id for r in report.results} == {"s1", "s2"}
    assert all(r.evaluator == "len_threshold" for r in report.results)


def test_runner_requires_model_when_prediction_missing():
    runner = BenchmarkRunner(model=None)
    with pytest.raises(ValueError):
        runner.run(_suite())


def test_runner_parallel_matches_sequential():
    seq = BenchmarkRunner(model=FakeModel(), max_workers=1).run(_suite())
    par = BenchmarkRunner(model=FakeModel(), max_workers=4).run(_suite())
    assert seq.pass_rate == par.pass_rate
    assert {r.sample_id for r in par.results} == {"s1", "s2"}


def test_evaluator_error_is_isolated():
    @register("boom")
    class Boom(Evaluator):
        name = "boom"

        def evaluate(self, sample):
            raise RuntimeError("kaboom")

    suite = SuiteConfig(
        name="err", evaluators=["boom"], samples=[EvalSample(id="x", prompt="p", prediction="y")]
    )
    report = BenchmarkRunner().run(suite)
    assert report.total == 1
    assert report.passed == 0
    assert report.results[0].details["status"] == "evaluator_error"


def test_markdown_render_contains_key_sections():
    report = BenchmarkRunner(model=FakeModel()).run(_suite())
    md = report.to_markdown()
    assert "# Crucible Report: unit" in md
    assert "## By evaluator" in md
    assert "## Per-sample results" in md
    assert "`len_threshold`" in md
    assert "Pass rate" in md
