"""CLI entry point — ingest files and ask questions."""

from typing import Optional

import typer

from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
app = typer.Typer(help="Hybrid-RAG CLI")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File path or YouTube video ID"),
    source_type: str = typer.Option("pdf", "--type", "-t", help="pdf|youtube|audio"),
    extractor: Optional[str] = typer.Option(
        None, "--extractor", "-e",
        help="Entity-extraction backend: 'local' (Ollama fine-tuned) or 'deepseek' "
             "(remote API). Default: EXTRACTION__BACKEND (deepseek).",
    ),
    store: bool = typer.Option(
        True, "--store/--no-store",
        help="Store phase: chunk + embed + write to pgvector. Disable to re-run "
             "only the graph phase against already-stored chunks (skips embedding).",
    ),
    graph: bool = typer.Option(
        True, "--graph/--no-graph",
        help="Graph phase: extract triplets and write to the Kùzu graph.",
    ),
):
    """Ingest a document into the RAG system.

    Two independent phases: the **store** phase (chunk → embed → pgvector) and the
    **graph** phase (entity/triplet extraction → Kùzu). Embedding is the slow,
    expensive part, so once it has run you can safely re-run just the graph phase
    (e.g. after a transient DeepSeek error) with `--no-store --graph`.
    """
    from ingestion.extractor import Extractor
    from ingestion.chunker import chunk_text, chunk_enrich
    from retrieval.pgvector import store_chunks, get_chunks_by_source
    from config.init_db import init_db
    from config.settings import get_settings
    from retrieval.kuzu_store import get_connection, init_graph_schema, upsert_triplets, link_entities_to_chunk
    from graph.entity_extraction import extract_entities_batch

    if not (store or graph):
        typer.echo("Nothing to do: both --no-store and --no-graph given.", err=True)
        raise typer.Exit(1)

    init_db()

    # Each item is (chunk_id, text, source_id) — the only fields the graph phase needs.
    graph_chunks: list[tuple[str, str, str]] = []

    if store:
        ext = Extractor()
        typer.echo(f"Extracting from {path} ({source_type})...")
        if source_type == "pdf":
            text = ext.extract_pdf(path)
        elif source_type == "youtube":
            text = ext.yt_subtitle_extraction(path)
        elif source_type == "audio":
            text = ext.reel_subtitle_extraction(path)
        else:
            typer.echo(f"Unknown source type: {source_type}", err=True)
            raise typer.Exit(1)

        typer.echo("Chunking and embedding...")
        chunks = chunk_enrich(chunk_text(text), source_type.upper(), path)
        stored = store_chunks(chunks)
        typer.echo(f"Stored {stored} chunks in pgvector.")
        graph_chunks = [(str(c.chunk_id), c.text, c.source_id) for c in chunks]

    if graph:
        if not graph_chunks:
            # Graph-only run: load already-stored chunks for this source.
            graph_chunks = get_chunks_by_source(path)
            if not graph_chunks:
                typer.echo(f"No stored chunks found for source {path!r}. "
                           f"Run the store phase first.", err=True)
                raise typer.Exit(1)
            typer.echo(f"Loaded {len(graph_chunks)} stored chunks for graph extraction.")

        # One Kùzu handle for the whole phase; `graph_db` kept alive so the
        # Database isn't GC'd while `graph_conn` is in use.
        graph_db, graph_conn = get_connection()
        init_graph_schema(conn=graph_conn)

        typer.echo(f"Extracting entities (backend: {extractor or 'default'})...")
        # Concurrency: extract a batch of chunks in parallel (I/O-bound API/LLM
        # calls), then write that batch's triplets to Kùzu serially (the Kùzu
        # connection is not thread-safe).
        s = get_settings()
        backend_name = (extractor or s.extraction.backend).lower()
        conc = s.ner.concurrency if backend_name in ("deepseek", "api") else s.extraction.concurrency

        total_triplets = 0
        failed = 0
        n = len(graph_chunks)
        for start in range(0, n, conc):
            batch = graph_chunks[start:start + conc]
            triplets_list = extract_entities_batch(
                [t for _, t, _ in batch], backend=extractor, max_workers=conc
            )
            for (chunk_id, text_, source_id), triplets in zip(batch, triplets_list):
                if triplets is None:  # extraction errored for this chunk
                    failed += 1
                    continue
                upsert_triplets(triplets, conn=graph_conn)
                entity_names = list({t.source.title for t in triplets} | {t.target.title for t in triplets})
                link_entities_to_chunk(entity_names, chunk_id, text=text_, source_id=source_id, conn=graph_conn)
                total_triplets += len(triplets)
            typer.echo(f"  processed {min(start + conc, n)}/{n} chunks (concurrency={conc})...")

        msg = f"Done. {total_triplets} triplets added to graph."
        if failed:
            msg += f" {failed} chunk(s) failed extraction (re-run with --no-store --graph)."
        typer.echo(msg)
    else:
        typer.echo("Done (store only; graph phase skipped).")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ask a question against the ingested knowledge base."""
    from reasoning.router import route_retrieval
    from retrieval.pgvector import cluster_routed_search
    from retrieval.kuzu_store import get_entity_context
    from retrieval.reranker import rerank
    from context.builder import build_context
    from llm.generator import generate
    from embeddings.embedder import embedder

    strategy = route_retrieval(question)
    if verbose:
        typer.echo(f"Strategy: {strategy['complexity']}")

    emb = embedder([question])[0]
    chunks = cluster_routed_search(emb)

    if not chunks:
        typer.echo("No relevant context found.")
        raise typer.Exit(1)

    # Stage 4: rerank the routed candidates down to the requested top-k.
    chunks = rerank(question, chunks, top_k=top_k)
    if verbose:
        typer.echo(f"Reranked to top {len(chunks)}.")

    graph_facts = ""
    if strategy["use_graph"]:
        entities = [w for w in question.split() if len(w) > 1 and w[0].isupper()]
        if entities:
            graph_facts = get_entity_context(entities)

    context, citations = build_context(chunks, graph_facts)
    answer = generate(question, context)

    typer.echo(f"\n{answer}\n")

    if verbose:
        typer.echo("\nSources:")
        for c in citations:
            typer.echo(f"  [{c.ref_id}] {c.source_id} — {c.text_preview}...")


@app.command()
def merge_graph(
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-s",
        help="Cosine similarity to treat entities as near-same (default: 0.90).",
    ),
    apply: bool = typer.Option(
        False, "--apply",
        help="Apply the merges. Without this flag it is a dry run (list only).",
    ),
):
    """Merge near-duplicate graph entities by embedding similarity.

    Run this *after* ingestion. By default it only lists the candidate pairs
    (the decision is yours); pass --apply to actually merge them.
    """
    from graph.merge import merge_similar_nodes

    candidates = merge_similar_nodes(threshold, apply=apply)
    if not candidates:
        typer.echo("No merge candidates found.")
        return

    verb = "Applied" if apply else "Found"
    typer.echo(f"{verb} {len(candidates)} merge candidate(s):")
    for c in candidates:
        typer.echo(f"  {c.drop!r} -> {c.keep!r}  (similarity={c.similarity:.3f})")

    if not apply:
        typer.echo("\nDry run — re-run with --apply to merge them.")


@app.command()
def reindex():
    """Cluster stored chunks (K-Means) and mark medoids for routed retrieval.

    An explicit, post-ingestion indexing step (like merge-graph). Recomputes
    cluster_id + is_medoid for every chunk; safe to re-run as the corpus grows.
    """
    from retrieval.cluster import reindex as do_reindex

    result = do_reindex()
    typer.echo(
        f"Reindexed {result.n_chunks} chunks into {result.k} clusters "
        f"({result.n_medoids} medoids)."
    )
    for cid in sorted(result.cluster_sizes):
        typer.echo(f"  cluster {cid}: {result.cluster_sizes[cid]} chunks")


if __name__ == "__main__":
    app()