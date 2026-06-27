"""Tests for report aggregation and JSON/Markdown serialisation."""

from __future__ import annotations

from pathlib import Path

from src.core.runner import BenchmarkRunner, FakeModel
from src.core.types import BenchmarkReport, EvalResult


def _report() -> BenchmarkReport:
    return BenchmarkReport(
        suite="demo",
        results=[
            EvalResult(score=1.0, passed=True, evaluator="a", sample_id="s1"),
            EvalResult(score=0.0, passed=False, evaluator="a", sample_id="s2"),
            EvalResult(score=0.5, passed=True, evaluator="b", sample_id="s1",
                       details={"status": "ok"}),
        ],
    )


def test_aggregate_stats():
    r = _report()
    assert r.total == 3
    assert r.passed == 2
    assert abs(r.pass_rate - 2 / 3) < 1e-9
    assert abs(r.mean_score - (1.0 + 0.0 + 0.5) / 3) < 1e-9


def test_by_evaluator_breakdown():
    r = _report()
    breakdown = r.by_evaluator()
    assert set(breakdown) == {"a", "b"}
    assert breakdown["a"]["total"] == 2
    assert breakdown["a"]["passed"] == 1
    assert breakdown["b"]["mean_score"] == 0.5


def test_json_roundtrip():
    r = _report()
    restored = BenchmarkReport.from_json(r.to_json())
    assert restored.suite == r.suite
    assert restored.total == r.total
    assert restored.pass_rate == r.pass_rate
    assert [x.sample_id for x in restored.results] == ["s1", "s2", "s1"]


def test_markdown_roundtrip_via_json(tmp_path: Path):
    r = _report()
    json_path = tmp_path / "run.json"
    json_path.write_text(r.to_json(), encoding="utf-8")

    restored = BenchmarkReport.from_json(json_path.read_text(encoding="utf-8"))
    md = restored.to_markdown()
    assert "# Crucible Report: demo" in md
    assert "66.7%" in md  # pass rate rendered
    assert "`a`" in md and "`b`" in md


def test_score_clamped():
    res = EvalResult(score=2.5, passed=True)
    assert res.score == 1.0
    res2 = EvalResult(score=-1.0, passed=False)
    assert res2.score == 0.0


def test_end_to_end_suite_file_runs(tmp_path: Path):
    # Build a tiny suite + dataset referencing the safety evaluator (no Docker).
    data = tmp_path / "data.jsonl"
    data.write_text(
        '{"id": "ok", "prompt": "p", "prediction": "Paris is the capital of France."}\n',
        encoding="utf-8",
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: tiny\nevaluators:\n  - safety\ndataset: data.jsonl\n", encoding="utf-8"
    )
    report = BenchmarkRunner(model=FakeModel()).run_suite(suite)
    assert report.suite == "tiny"
    assert report.total == 1
    assert report.results[0].evaluator == "safety"
