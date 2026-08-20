"""Evaluation CLI — prepare data, ingest the corpus, run the staged benchmark.

The benchmark runs in decoupled, cache-persisted phases (see
``evaluation/README.md``):

    artifacts  -> 1 batched embedding call + 1 bundled entity call (all queries)
    retrieve   -> every (config x query) cell; final reranked chunks persisted
    generate   -> bundled generation (~40 pairs/call) from persisted retrievals
    report     -> score + markdown

Run from the repo root (the ``--extra eval`` installs the `datasets` dep):

    uv run --extra eval python -m evaluation.run_eval prepare
    uv run --extra eval python -m evaluation.run_eval ingest --with-graph
    uv run --extra eval python -m evaluation.run_eval artifacts
    uv run --extra eval python -m evaluation.run_eval retrieve
    uv run --extra eval python -m evaluation.run_eval generate
    uv run --extra eval python -m evaluation.run_eval report
    uv run --extra eval python -m evaluation.run_eval all --with-graph   # everything

Every phase resumes from its JSON cache and re-runs only on ``--force``.
`prepare` only needs the `datasets` dependency. The other commands need the live
services (PostgreSQL + pgvector, Mistral embeddings, Jina rerank, Gemini NER +
generation), exactly like the main app.
"""

from __future__ import annotations

import typer

from evaluation import config, dataset, persist
from evaluation import report as report_mod
from evaluation import runner as runner_mod

app = typer.Typer(help="Hybrid-RAG HotpotQA evaluation harness", no_args_is_help=True)


def _load_records(n: int, seed: int) -> list[dict]:
    return dataset.load_cached(n=n, seed=seed)


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
def artifacts(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    force: bool = typer.Option(False, "--force", help="Recompute this phase."),
):
    """Phase 1: query embeddings (1 batched call) + query entities (1 bundled call)."""
    records = _load_records(n, seed)
    typer.echo(
        f"Computing artifacts for {len(records)} queries (seed={seed})... "
        "(1 Mistral embedding call + 1 Gemini entity call)"
    )
    embeddings, entities = runner_mod.artifacts(records, n=n, seed=seed, force=force)
    typer.echo(
        f"Done: embeddings cached for {len(embeddings)} queries, entities for "
        f"{len(entities)} queries."
    )


@app.command()
def retrieve(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    candidate_k: int = typer.Option(config.CANDIDATE_K, "--candidate-k"),
    force: bool = typer.Option(False, "--force", help="Recompute this phase."),
):
    """Phase 2: run all retrieval cells; persist final reranked chunks (resumable)."""
    records = _load_records(n, seed)
    embeddings, entities = runner_mod.artifacts(records, n=n, seed=seed)

    state = {"last": ""}

    def _progress(label: str, done: int, total: int):
        if label != state["last"] or done == total:
            state["last"] = label
            typer.echo(f"  [{label}] {done}/{total}", err=True)

    typer.echo(f"Running retrieval over {len(records)} queries (top_k={top_k})...")
    retrievals = runner_mod.retrieve(
        records,
        embeddings,
        entities,
        top_k=top_k,
        candidate_k=candidate_k,
        n=n,
        seed=seed,
        force=force,
        on_progress=_progress,
    )
    cells = sum(len(v) for v in retrievals.values())
    typer.echo(f"Done: {cells} (config x query) cells persisted.")

    if not force and _missing_retrieval_cells(records, retrievals):
        typer.echo("Note: some cells were skipped from cache; pass --force to recompute.")


def _missing_retrieval_cells(records: list[dict], retrievals: dict) -> bool:
    specs = [s.label for s in runner_mod.build_run_specs()]
    for record in records:
        cells = retrievals.get(record["id"], {})
        for label in specs:
            if label != "direct" and label not in cells:
                return True
    return False


@app.command()
def generate(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    batch_size: int = typer.Option(
        config.GENERATION_BATCH_SIZE, "--batch-size", help="(question, context) pairs per call."
    ),
    force: bool = typer.Option(False, "--force", help="Recompute this phase."),
):
    """Phase 3: bundled generation from persisted retrievals (zero retrieval work)."""
    records = _load_records(n, seed)
    retrievals = persist.load("retrievals", n, seed) or {}
    if not retrievals:
        typer.echo("No retrievals cached — run `retrieve` first.", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"Generating answers from {sum(len(v) for v in retrievals.values())} cached "
        f"retrieval cells (batch_size={batch_size}, ~{len(records) // batch_size + 1} calls/config)..."
    )
    answers = runner_mod.generate(records, retrievals, batch_size=batch_size, n=n, seed=seed, force=force)
    typer.echo(f"Done: answers cached for {len(answers)} queries x configs.")


@app.command()
def report(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    graph_ingested: bool = typer.Option(
        True,
        "--graph-ingested/--no-graph-ingested",
        help="Whether the graph was ingested (annotates the report only).",
    ),
    stem: str | None = typer.Option(None, "--stem", help="Report filename stem."),
):
    """Phase 4: score from caches and write the markdown report (+ raw JSON)."""
    records = _load_records(n, seed)
    retrievals = persist.load("retrievals", n, seed) or {}
    answers = persist.load("answers", n, seed) or {}
    if not retrievals or not answers:
        typer.echo("Missing phases — run `retrieve` and `generate` first.", err=True)
        raise typer.Exit(1)

    results = runner_mod.score(records, retrievals, answers, top_k=top_k)
    out = runner_mod.report(
        records,
        results,
        top_k=top_k,
        graph_ingested=graph_ingested,
        seed=seed,
        stem=stem,
    )
    typer.echo("\n" + report_mod.to_markdown(out["results"], {"n": len(records), "seed": seed, "top_k": top_k, "graph_ingested": graph_ingested, "staged": True}))
    typer.echo(f"Report written: {out['md'].name}  (raw JSON: {out['json'].name})")


@app.command()
def all(
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    candidate_k: int = typer.Option(config.CANDIDATE_K, "--candidate-k"),
    batch_size: int = typer.Option(
        config.GENERATION_BATCH_SIZE, "--batch-size", help="(question, context) pairs per call."
    ),
    with_graph: bool = typer.Option(False, "--with-graph/--no-graph"),
    backend: str | None = typer.Option(None, "--extractor", "-e"),
    force: bool = typer.Option(False, "--force", help="Recompute every phase."),
):
    """Convenience: prepare -> ingest -> artifacts -> retrieve -> generate -> report."""
    from evaluation.corpus import ingest_corpus

    records = dataset.prepare_dataset(n=n, seed=seed)
    corpus = dataset.build_corpus(records)
    typer.echo(
        f"Prepared {len(records)} queries; ingesting {len(corpus)} paragraphs "
        f"(graph={'on' if with_graph else 'off'})..."
    )
    ingest_corpus(corpus, with_graph=with_graph, backend=backend)

    runner_mod.artifacts(records, n=n, seed=seed, force=force)
    retrievals = runner_mod.retrieve(
        records,
        *runner_mod.artifacts(records, n=n, seed=seed),
        top_k=top_k,
        candidate_k=candidate_k,
        n=n,
        seed=seed,
        force=force,
    )
    answers = runner_mod.generate(records, retrievals, batch_size=batch_size, n=n, seed=seed, force=force)

    results = runner_mod.score(records, retrievals, answers, top_k=top_k)
    out = runner_mod.report(
        records,
        results,
        top_k=top_k,
        graph_ingested=with_graph,
        seed=seed,
    )
    typer.echo("\n" + report_mod.to_markdown(out["results"], {"n": len(records), "seed": seed, "top_k": top_k, "graph_ingested": with_graph, "staged": True}))
    typer.echo(f"Report written: {out['md'].name}")


@app.command()
def clear(
    kind: str | None = typer.Option(
        None,
        "--kind",
        help="Phase file to clear (query_embeddings|query_entities|retrievals|answers); "
        "omit to wipe the whole cache.",
    ),
    n: int = typer.Option(config.N_QUERIES, "--n"),
    seed: int = typer.Option(config.SEED, "--seed"),
):
    """Wipe staged-phase cache files (or the whole eval_cache dir)."""
    removed = persist.clear(kind, n, seed)
    target = f"phase {kind!r}" if kind else "whole cache"
    typer.echo(f"Cleared {target}: {removed} file(s) removed.")


if __name__ == "__main__":
    app()