"""Crucible command-line interface.

Commands:
    * ``crucible run <suite.yaml>`` — run a suite and write JSON + Markdown.
    * ``crucible list-evaluators``   — show registered evaluators.
    * ``crucible report <run.json>`` — re-render a saved JSON report as Markdown.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from src.core.registry import get_evaluator_class, list_evaluators
from src.core.runner import BenchmarkRunner, echo_model
from src.core.types import BenchmarkReport

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="crucible",
    help="GuildLM Crucible — evaluate specialist models with pluggable evaluators.",
    no_args_is_help=True,
    add_completion=False,
)


def _configure_logging(verbose: bool) -> None:
    """Set up root logging at INFO (or DEBUG when verbose)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def run(
    suite: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to a suite YAML."),
    output_dir: Path = typer.Option(
        Path("reports"), "--output-dir", "-o", help="Directory for JSON/Markdown reports."
    ),
    workers: int = typer.Option(1, "--workers", "-w", min=1, help="Parallel evaluation workers."),
    echo: bool = typer.Option(
        False, "--echo", help="Use the built-in echo model for missing predictions."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run a suite and write ``<suite>.json`` and ``<suite>.md`` reports."""
    _configure_logging(verbose)
    runner = BenchmarkRunner(model=echo_model if echo else None, max_workers=workers)
    report = runner.run_suite(suite)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = suite.stem
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    typer.echo(
        f"Suite '{report.suite}': pass rate {report.pass_rate:.1%} "
        f"({report.passed}/{report.total}), mean score {report.mean_score:.3f}"
    )
    typer.echo(f"Wrote {json_path} and {md_path}")


@app.command("list-evaluators")
def list_evaluators_cmd() -> None:
    """List every registered evaluator and its docstring summary."""
    names = list_evaluators()
    if not names:
        typer.echo("No evaluators registered.")
        raise typer.Exit(code=0)
    for name in names:
        cls = get_evaluator_class(name)
        summary = (cls.__doc__ or "").strip().splitlines()
        first_line = summary[0] if summary else ""
        typer.echo(f"{name:<16} {first_line}")


@app.command()
def report(
    run_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Saved run JSON."),
    output: Path = typer.Option(None, "--output", "-o", help="Write Markdown here instead of stdout."),
) -> None:
    """Re-render a saved JSON report as Markdown."""
    rep = BenchmarkReport.from_json(run_json.read_text(encoding="utf-8"))
    markdown = rep.to_markdown()
    if output:
        output.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(markdown)


if __name__ == "__main__":  # pragma: no cover
    app()
