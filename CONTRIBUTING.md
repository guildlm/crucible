# Contributing to Crucible

Thanks for helping improve GuildLM Crucible. This guide covers the local setup,
quality bar and the workflow for adding evaluators.

## Local setup

```bash
python -m venv .venv
.venv/bin/pip install typer pyyaml openai pytest
.venv/bin/python -m pytest -q
```

The test suite must pass with **only** `typer`, `pyyaml`, `openai` and
`pytest` installed — **no Docker and no network**. Any code path that needs
Docker or a live endpoint must be mockable and have an offline fallback.

## Quality bar

- **Type hints** on all public functions and methods.
- **Docstrings** on every module, public class and function.
- **Logging**, not `print`, for diagnostics (`logging.getLogger(__name__)`).
- **No dead code** and no committed secrets.
- New behaviour ships with tests. Keep coverage of pass *and* fail paths.

## Adding an evaluator

1. Create a module under `src/evaluators/`.
2. Subclass `src.core.registry.Evaluator`, set `name`, implement
   `evaluate(self, sample) -> EvalResult`.
3. Decorate the class with `@register("<name>")`.
4. Add the module path to `_BUILTIN_MODULES` in `src/core/registry.py` so it is
   auto-discovered.
5. Accept configuration as **keyword-only** constructor args (these are
   supplied from a suite's `config:` block).
6. Add tests under `tests/`. If the evaluator executes code, hide that behind
   the `Sandbox` interface and test with a fake sandbox.

## Commits & PRs

- Keep commits focused and descriptive.
- Ensure `pytest -q` is green before opening a PR.
- CI (`.github/workflows/ci.yml`) runs the suite on Python 3.10–3.12.

By contributing you agree your contributions are licensed under Apache-2.0.
