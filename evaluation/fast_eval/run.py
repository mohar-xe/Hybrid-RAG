"""fast_eval CLI: check → (ingest if needed) → run → report.

Run from the repo root:

    uv run python -m evaluation.fast_eval.run

Convenience flags mirror the staged CLI (``--n``, ``--seed``, ``--top-k``,
``--candidate-k``, ``--workers``). The provider is configured via
``evaluation/fast_eval/config.py`` (or ``FAST_EVAL__*`` env vars), not here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    import typer

    from evaluation.fast_eval import config
    from evaluation.fast_eval.check_ingest import check_ingested, ingest_if_missing
    from evaluation.fast_eval.report import build_results, write_report
    from evaluation.fast_eval.runner import run

    app = typer.Typer(help="fast_eval — concurrent, paid, real-latency evaluation")

    @app.command()
    def go(
        n: int = typer.Option(config.N_QUERIES, "--n"),
        seed: int = typer.Option(config.SEED, "--seed"),
        top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
        candidate_k: int = typer.Option(config.CANDIDATE_K, "--candidate-k"),
        workers: int = typer.Option(config.MAX_WORKERS, "--workers"),
        with_graph: bool = typer.Option(False, "--with-graph"),
    ) -> None:

        # Apply model/provider config to the live environment before importing
        # settings-dependent modules.
        os.environ.setdefault("GENERATOR__MODEL", config.GENERATOR_MODEL)
        os.environ.setdefault("GENERATOR__BASE_URL", config.GENERATOR_BASE_URL)
        os.environ.setdefault("GENERATOR__FALLBACK_ENABLED", str(config.GENERATOR_FALLBACK_ENABLED).lower())
        if config.GENERATOR_MAX_TOKENS:
            os.environ["GENERATOR__MAX_TOKENS"] = str(config.GENERATOR_MAX_TOKENS)
        # Entity extraction (query NER) is paid-only here too: never fall back to
        # the slow local Ollama model. A transient API failure is retried by
        # fast_eval's own backoff, then degrades to no graph seeding.
        os.environ["NER__FALLBACK_ENABLED"] = "false"
        key = config.resolve_api_key()
        if key:
            os.environ["GENERATOR__API_KEY"] = key

        records = _load_records(n, seed)

        # 1. Ingestion check (skip if the corpus is already present).
        corpus = _build_corpus(records)
        state = check_ingested(config.CORPUS_SOURCE_TYPE)
        if state.vector_ready:
            typer.echo("Corpus already ingested — skipping to retrieval.")
        elif config.INGEST_IF_MISSING:
            typer.echo("Corpus missing — ingesting before retrieval...")
            ingest_if_missing(corpus, with_graph=with_graph)
        else:
            typer.echo("Corpus missing and INGEST_IF_MISSING=False — aborting.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Running full ablation over {len(records)} queries "
                   f"(workers={workers}, top_k={top_k})...")

        cells, tracker = run(
            records,
            top_k=top_k,
            candidate_k=candidate_k,
            max_workers=workers,
            on_spec_start=lambda label, total: typer.echo(
                f"  [{label}] 0/{total}", err=True
            ),
            on_cell=lambda label, done, total: typer.echo(
                f"  [{label}] {done}/{total}", err=True
            ),
        )

        rows = build_results(cells)
        from datetime import datetime, timezone

        meta = {
            "n": len(records),
            "seed": seed,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "max_workers": workers,
            "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            **tracker.summary(),
        }
        md_path, json_path = write_report(rows, meta)
        typer.echo("\n" + _render(rows, meta))
        typer.echo(f"Report written: {md_path.name}  (raw JSON: {json_path.name})")

    def _render(rows, meta):
        from evaluation.fast_eval.report import to_markdown
        return to_markdown(rows, meta)

    app()


def _load_records(n: int, seed: int):
    from evaluation.dataset import load_cached, prepare_dataset

    try:
        return load_cached(n=n, seed=seed)
    except FileNotFoundError:
        return prepare_dataset(n=n, seed=seed)


def _build_corpus(records):
    corpus: dict[str, str] = {}
    for r in records:
        for p in r["paragraphs"]:
            if p["title"] not in corpus and p["text"]:
                corpus[p["title"]] = p["text"]
    return corpus


if __name__ == "__main__":
    main()
