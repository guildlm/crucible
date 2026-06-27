"""Tests for the evaluator registry."""

from __future__ import annotations

import pytest

from src.core.registry import (
    Evaluator,
    get_evaluator,
    get_evaluator_class,
    list_evaluators,
    register,
)
from src.core.types import EvalResult, EvalSample


def test_builtins_are_discoverable():
    names = list_evaluators()
    assert {"go_functional", "llm_judge", "safety"} <= set(names)


def test_register_and_instantiate():
    @register("unit_dummy")
    class Dummy(Evaluator):
        name = "unit_dummy"

        def __init__(self, *, weight: float = 1.0) -> None:
            self.weight = weight

        def evaluate(self, sample: EvalSample) -> EvalResult:
            return EvalResult(score=self.weight, passed=True)

    assert get_evaluator_class("unit_dummy") is Dummy
    ev = get_evaluator("unit_dummy", weight=0.5)
    assert ev.weight == 0.5
    result = ev.evaluate(EvalSample(id="x", prompt="p"))
    assert result.score == 0.5 and result.passed


def test_duplicate_registration_rejected():
    @register("unit_dup")
    class A(Evaluator):
        name = "unit_dup"

        def evaluate(self, sample):  # pragma: no cover - not called
            return EvalResult(score=1.0, passed=True)

    with pytest.raises(ValueError):

        @register("unit_dup")
        class B(Evaluator):
            name = "unit_dup"

            def evaluate(self, sample):  # pragma: no cover - not called
                return EvalResult(score=0.0, passed=False)


def test_unknown_evaluator_raises():
    with pytest.raises(KeyError):
        get_evaluator_class("does_not_exist")


def test_register_requires_evaluator_subclass():
    with pytest.raises(TypeError):

        @register("not_an_evaluator")
        class Plain:  # noqa: D401 - intentionally not an Evaluator
            pass
