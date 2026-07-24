"""Evaluation CLI — prepare data, ingest the corpus, run the benchmark, report.

Run from the repo root (the ``--extra eval`` installs the `datasets` dep):

    uv run --extra eval python -m evaluation.run_eval prepare
    uv run --extra eval python -m evaluation.run_eval ingest --with-graph
    uv run --extra eval python -m evaluation.run_eval run
    uv run --extra eval python -m evaluation.run_eval all --with-graph   # all of the above

`prepare` only needs the `datasets` dependency. `ingest` and `run` need the live
services (PostgreSQL + pgvector, Ollama for embeddings, and the configured
generator endpoint), exactly like the main app.
"""

from __future__ import annotations

import typer

from evaluation import config, dataset, report
from evaluation.runner import run_all

app = typer.Typer(help="Hybrid-RAG HotpotQA evaluation harness", no_args_is_help=True)


@app.command()
def prepare(
    n: int = typer.Option(config.N_QUERIES, "--n", help="Number of queries to select."),
    seed: int = typer.Option(config.SEED, "--seed", help="RNG seed (reproducibility)."),
    force: bool = typer.Option(
        False, "--force", help="Re-download and overwrite the cache."
    ),
):
    """Download HotpotQA and cache the seeded query selection to JSON."""
    records = dataset.prepare_dataset(n=n, seed=seed, force=force)
    corpus = dataset.build_corpus(records)
    typer.echo(
        f"Prepared {len(records)} queries (seed={seed}); corpus has "
        f"{len(corpus)} unique paragraphs. Cached to {config.selection_cache_path(n, seed)}."
    )


@app.command()
def ingest(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    with_graph: bool = typer.Option(
        False,
        "--with-graph/--no-graph",
        help="Also extract the knowledge graph (needed for `all_three` graph facts). Slow.",
    ),
    backend: str | None = typer.Option(
        None, "--extractor", "-e", help="Graph extractor backend."
    ),
):
    """Ingest the selected HotpotQA paragraphs into pgvector (+ optional graph)."""
    from evaluation.corpus import ingest_corpus

    records = dataset.load_cached(n=n, seed=seed)
    corpus = dataset.build_corpus(records)
    typer.echo(
        f"Ingesting {len(corpus)} paragraphs (graph={'on' if with_graph else 'off'})..."
    )
    counts = ingest_corpus(corpus, with_graph=with_graph, backend=backend)
    typer.echo(
        f"Done: {counts['titles']} paragraphs, {counts['chunks']} chunks, "
        f"{counts['graph_triplets']} triplets ({counts['graph_failed']} failures)."
    )


@app.command()
def run(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    candidate_k: int = typer.Option(config.CANDIDATE_K, "--candidate-k"),
    graph_ingested: bool = typer.Option(
        True,
        "--graph-ingested/--no-graph-ingested",
        help="Whether the graph was ingested (annotates the report only).",
    ),
):
    """Run the full comparison matrix and write the markdown report."""
    records = dataset.load_cached(n=n, seed=seed)

    state = {"last": ""}

    def _progress(label: str, done: int, total: int):
        if label != state["last"] or done == total:
            state["last"] = label
            typer.echo(f"  [{label}] {done}/{total}", err=True)

    typer.echo(
        f"Running evaluation over {len(records)} queries (seed={seed}, top_k={top_k})..."
    )
    results = run_all(
        records, top_k=top_k, candidate_k=candidate_k, on_progress=_progress
    )
    payload = [r.to_dict() for r in results]

    meta = {
        "n": len(records),
        "seed": seed,
        "top_k": top_k,
        "graph_ingested": graph_ingested,
    }
    md_path = report.write_report(payload, config.RESULTS_DIR, meta=meta)

    typer.echo("\n" + report.to_markdown(payload, meta))
    typer.echo(f"Report written: {md_path.name}")


@app.command()
def all(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    candidate_k: int = typer.Option(config.CANDIDATE_K, "--candidate-k"),
    with_graph: bool = typer.Option(False, "--with-graph/--no-graph"),
    backend: str | None = typer.Option(None, "--extractor", "-e"),
):
    """Convenience: prepare -> ingest -> run, end to end."""
    from evaluation.corpus import ingest_corpus

    records = dataset.prepare_dataset(n=n, seed=seed)
    corpus = dataset.build_corpus(records)
    typer.echo(
        f"Prepared {len(records)} queries; ingesting {len(corpus)} paragraphs..."
    )
    ingest_corpus(corpus, with_graph=with_graph, backend=backend)

    typer.echo("Running evaluation...")
    results = run_all(records, top_k=top_k, candidate_k=candidate_k)
    payload = [r.to_dict() for r in results]
    meta = {
        "n": len(records),
        "seed": seed,
        "top_k": top_k,
        "graph_ingested": with_graph,
    }
    md_path = report.write_report(payload, config.RESULTS_DIR, meta=meta)
    typer.echo("\n" + report.to_markdown(payload, meta))
    typer.echo(f"Report written: {md_path.name}")


if __name__ == "__main__":
    app()
