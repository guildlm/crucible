"""Pluggable evaluator registry.

Evaluators implement the :class:`Evaluator` interface and register themselves
with the :func:`register` decorator. The :class:`BenchmarkRunner` and the CLI
resolve evaluators by name through :func:`get_evaluator`.

Example:
    >>> from src.core.registry import register, Evaluator
    >>> @register("noop")
    ... class NoOp(Evaluator):
    ...     name = "noop"
    ...     def evaluate(self, sample):
    ...         from src.core.types import EvalResult
    ...         return EvalResult(score=1.0, passed=True)
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from src.core.types import EvalResult, EvalSample

logger = logging.getLogger(__name__)

__all__ = [
    "Evaluator",
    "register",
    "get_evaluator",
    "get_evaluator_class",
    "list_evaluators",
    "discover_builtins",
]


class Evaluator(ABC):
    """Abstract base class every evaluator must implement.

    Subclasses set a class-level ``name`` and implement :meth:`evaluate`.
    Construction is expected to accept keyword configuration only, so the
    runner can instantiate any evaluator uniformly from a suite config.
    """

    #: Unique registry name. Set by subclasses (and/or the ``@register`` call).
    name: str = ""

    @abstractmethod
    def evaluate(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single sample and return a structured result."""
        raise NotImplementedError


# Internal registry: name -> Evaluator subclass.
_REGISTRY: dict[str, type[Evaluator]] = {}

# Built-in evaluator modules eagerly imported by :func:`discover_builtins`.
_BUILTIN_MODULES: tuple[str, ...] = (
    "src.evaluators.go_eval",
    "src.evaluators.quality_eval",
    "src.evaluators.safety_eval",
)


def register(name: str):
    """Class decorator registering an :class:`Evaluator` under ``name``.

    Args:
        name: Unique key used to look the evaluator up later.

    Raises:
        ValueError: If ``name`` is empty or already registered to a different
            class.
    """

    if not name:
        raise ValueError("Evaluator name must be a non-empty string.")

    def decorator(cls: type[Evaluator]) -> type[Evaluator]:
        if not issubclass(cls, Evaluator):
            raise TypeError(f"{cls!r} must subclass Evaluator to be registered.")
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Evaluator name {name!r} already registered to {existing.__name__}."
            )
        cls.name = name
        _REGISTRY[name] = cls
        logger.debug("Registered evaluator %r -> %s", name, cls.__name__)
        return cls

    return decorator


def get_evaluator_class(name: str) -> type[Evaluator]:
    """Return the registered evaluator class for ``name``.

    Raises:
        KeyError: If no evaluator is registered under ``name``.
    """

    discover_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No evaluator registered as {name!r}. "
            f"Known evaluators: {sorted(_REGISTRY)}"
        ) from None


def get_evaluator(name: str, **config: Any) -> Evaluator:
    """Instantiate the evaluator registered as ``name`` with ``config``.

    Args:
        name: Registry key of the evaluator.
        **config: Keyword arguments forwarded to the evaluator constructor.
    """

    cls = get_evaluator_class(name)
    return cls(**config)


def list_evaluators() -> list[str]:
    """Return the sorted names of all registered evaluators."""

    discover_builtins()
    return sorted(_REGISTRY)


def discover_builtins() -> None:
    """Import built-in evaluator modules so their registrations run.

    Idempotent: importing an already-imported module is a cheap no-op. Import
    failures are logged but never raised, so a missing optional dependency in
    one evaluator cannot break the whole registry.
    """

    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.warning("Could not import built-in evaluator %s: %s", module, exc)
