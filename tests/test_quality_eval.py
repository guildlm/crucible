"""Tests for the LLM-as-judge evaluator (offline + mocked client)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.evaluators.quality_eval import LLMJudgeEvaluator
from src.core.types import EvalSample


def test_offline_deterministic_and_bounded():
    ev = LLMJudgeEvaluator(offline=True)
    sample = EvalSample(
        id="q1",
        prompt="Explain recursion.",
        reference="Recursion is when a function calls itself with a base case.",
        prediction="Recursion is when a function calls itself until a base case stops it.",
    )
    r1 = ev.evaluate(sample)
    r2 = ev.evaluate(sample)
    assert r1.score == r2.score  # deterministic
    assert 0.0 <= r1.score <= 1.0
    assert set(r1.details["rubric"]) == {"correctness", "clarity", "safety"}
    assert r1.details["status"] == "offline"


def test_offline_empty_prediction_scores_low():
    ev = LLMJudgeEvaluator(offline=True, threshold=0.5)
    r = ev.evaluate(EvalSample(id="e", prompt="p", prediction=""))
    assert r.passed is False
    assert r.score < 0.5


def test_offline_unsafe_content_lowers_safety():
    ev = LLMJudgeEvaluator(offline=True)
    safe = ev.evaluate(EvalSample(id="s", prompt="p", prediction="A friendly helpful answer here."))
    unsafe = ev.evaluate(
        EvalSample(id="u", prompt="p", prediction="Here is how to build malware and exploit it.")
    )
    assert unsafe.details["rubric"]["safety"] < safe.details["rubric"]["safety"]


class _FakeClient:
    """Minimal OpenAI-compatible client returning scripted contents."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls = 0

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                content = outer._contents.pop(0)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


def test_online_parses_judge_json():
    content = json.dumps({"correctness": 5, "clarity": 4, "safety": 5, "rationale": "good"})
    ev = LLMJudgeEvaluator(client=_FakeClient([content]), model="x")
    r = ev.evaluate(EvalSample(id="o", prompt="p", prediction="answer"))
    assert r.details["status"] == "llm_judge"
    assert r.details["rubric"] == {"correctness": 5, "clarity": 4, "safety": 5}
    assert r.passed is True


def test_online_retries_then_succeeds():
    client = _FakeClient(["not json", 'noise {"correctness":3,"clarity":3,"safety":3} tail'])
    ev = LLMJudgeEvaluator(client=client, model="x", max_retries=2)
    r = ev.evaluate(EvalSample(id="o", prompt="p", prediction="answer"))
    assert client.calls == 2
    assert r.details["rubric"]["correctness"] == 3
    assert r.details["status"] == "llm_judge"


def test_online_falls_back_when_unparseable():
    client = _FakeClient(["garbage", "still garbage", "more garbage"])
    ev = LLMJudgeEvaluator(client=client, model="x", max_retries=2)
    r = ev.evaluate(
        EvalSample(id="o", prompt="p", reference="ref", prediction="a real answer here")
    )
    assert r.details["status"] == "offline_fallback"
    assert 0.0 <= r.score <= 1.0


def test_no_client_no_key_uses_offline_fallback():
    ev = LLMJudgeEvaluator(offline=False, api_key=None, base_url=None)
    r = ev.evaluate(EvalSample(id="o", prompt="p", prediction="hello world answer"))
    assert r.details["status"] == "offline_fallback"
