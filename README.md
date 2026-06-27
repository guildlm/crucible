# Crucible

**Crucible** is GuildLM's evaluation framework. It measures how well GuildLM
specialist models perform across *any* domain through small, composable,
**pluggable evaluators** — functional tests, factual accuracy, quality scoring,
safety/hallucination heuristics, and LLM-as-judge — and renders the results as
both machine-readable JSON and a reviewable Markdown report.

See [GuildLM](https://github.com/guildlm/guildlm.github.io) for the full
architecture; Crucible is the component that turns "does this model actually
work?" into a number you can track over time.

---

## Why Crucible

- **Pluggable.** Every check is an `Evaluator` registered by name. Add a new
  one with a decorator; the runner and CLI discover it automatically.
- **Backend-agnostic.** Inject any `prompt -> completion` callable to evaluate
  any model or API. A deterministic `FakeModel` keeps runs reproducible.
- **Safe by construction.** Code execution happens inside a locked-down Docker
  sandbox (no network, capped CPU/memory/PIDs, wall-clock timeout) behind a
  mockable interface — so CI runs green with **no Docker and no network**.
- **Reportable.** Aggregate stats, per-evaluator breakdown and per-sample
  detail, serialisable to JSON and re-renderable to Markdown.

## Install

```bash
python -m venv .venv
.venv/bin/pip install typer pyyaml openai pytest
# or, as a package:
.venv/bin/pip install -e .
```

## Quick start

```bash
# List the evaluators available in this build
crucible list-evaluators

# Run a suite -> writes reports/<suite>.json and reports/<suite>.md
crucible run suites/go_basic.yaml

# Re-render a saved JSON run as Markdown
crucible report reports/go_basic.json
```

Programmatic use:

```python
from src.core.runner import BenchmarkRunner, FakeModel

runner = BenchmarkRunner(model=FakeModel(), max_workers=4)
report = runner.run_suite("suites/go_basic.yaml")
print(report.to_markdown())
print(f"pass rate: {report.pass_rate:.1%}")
```

## Suite format

A suite is a YAML file pointing at a JSONL dataset:

```yaml
name: go_basic
description: Basic Go functional checks
evaluators:
  - go_functional        # one or more registered evaluator names
dataset: data/go_basic.jsonl   # path relative to the suite file
config:                  # optional per-evaluator constructor kwargs
  go_functional:
    timeout: 30.0
```

Each dataset line is one sample:

```json
{"id": "add", "prompt": "Write Add(a,b int) int", "reference": "...",
 "prediction": "package sandbox\nfunc Add(a,b int) int { return a+b }",
 "metadata": {"tests": "package sandbox\nimport \"testing\"\n..."}}
```

`prediction` is optional — if omitted, the runner calls the injected model with
`prompt` to generate it. Evaluators read whatever they need from `metadata`.

## Evaluator catalog

| Name | Purpose | Key inputs (`metadata`) | Needs |
| --- | --- | --- | --- |
| `go_functional` | `go vet` + `go build`/`go test` in a sandbox | `tests`, `module` | Docker (mockable) |
| `llm_judge` | Rubric LLM-as-judge: correctness / clarity / safety | — | OpenAI-compatible endpoint, or `offline: true` |
| `safety` | Refusal correctness, banned-pattern, citation presence | `expect_refusal`, `require_citation`, `banned_patterns` | pure Python |

### `go_functional`

Materialises `main.go` + `main_test.go` + `go.mod` and runs `go vet ./...`
then `go test ./...` inside the sandbox. Passes only when both succeed; vet and
build/test failures are reported distinctly. Falls back to `go build` when no
tests are supplied. When Docker is unavailable it returns a clear
`status: "unavailable"` rather than crashing.

### `llm_judge`

Scores predictions on a 1–5 rubric per dimension via an OpenAI-compatible chat
endpoint, robustly parsing the JSON verdict with retries. Configure through the
environment:

```bash
export CRUCIBLE_JUDGE_BASE_URL="https://api.openai.com/v1"
export CRUCIBLE_JUDGE_API_KEY="sk-..."
export CRUCIBLE_JUDGE_MODEL="gpt-4o-mini"
```

Set `offline: true` (in suite `config`) for a deterministic, network-free
heuristic mode used in tests and CI.

### `safety`

Pure-Python hallucination/safety heuristics: detects refusals (and checks them
against the sample's `expect_refusal` expectation), scans for banned regex
patterns (defaults + per-sample), and enforces citation presence when
`require_citation` is set. Any banned-content hit is a hard fail.

## Writing a custom evaluator

```python
from src.core.registry import Evaluator, register
from src.core.types import EvalResult, EvalSample

@register("keyword_match")
class KeywordMatch(Evaluator):
    """Pass when the prediction contains an expected keyword."""

    name = "keyword_match"

    def __init__(self, *, keyword: str = "") -> None:
        self.keyword = keyword

    def evaluate(self, sample: EvalSample) -> EvalResult:
        hit = self.keyword.lower() in (sample.prediction or "").lower()
        return EvalResult(score=1.0 if hit else 0.0, passed=hit,
                          details={"keyword": self.keyword})
```

Place it under `src/evaluators/`, add the module to `_BUILTIN_MODULES` in
`src/core/registry.py` (or import it yourself), and reference it by name in a
suite. Constructor kwargs come from the suite's `config:` block.

## Sandbox security notes

`go_functional` never runs untrusted code on the host. `DockerSandbox`
(`src/core/sandbox.py`) launches each command with:

- `--network none` — no inbound or outbound network.
- `--memory 512m`, `--cpus 1.0`, `--pids-limit 256` — resource ceilings.
- `--read-only` root filesystem with a single writable workspace bind mount and
  an `exec` tmpfs for `/tmp`.
- `GOPROXY=off` — no module downloads.
- `--rm` plus a host-enforced wall-clock `timeout` as a backstop.

The Docker call sits behind the `Sandbox` interface, so tests inject a
`FakeSandbox` and the entire suite runs without Docker. When the daemon is
absent, evaluation fails *gracefully* with `status: "unavailable"`.

## Sample report

```markdown
# Crucible Report: go_basic

**Pass rate:** 100.0% (2/2) &nbsp; **Mean score:** 1.000

## By evaluator

| Evaluator | Passed | Total | Pass rate | Mean score |
| --- | ---: | ---: | ---: | ---: |
| `go_functional` | 2 | 2 | 100.0% | 1.000 |

## Per-sample results

| Sample | Evaluator | Result | Score | Notes |
| --- | --- | :---: | ---: | --- |
| `add` | `go_functional` | ✅ pass | 1.000 | ok |
| `reverse` | `go_functional` | ✅ pass | 1.000 | ok |
```

## Development

```bash
python -m venv .venv
.venv/bin/pip install typer pyyaml openai pytest
.venv/bin/python -m pytest -q
```

The full test suite runs with **no Docker and no network**. See
[CONTRIBUTING.md](CONTRIBUTING.md). Licensed under Apache-2.0.
