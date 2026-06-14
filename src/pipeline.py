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
):
    """Ingest a document into the RAG system."""
    from ingestion.extractor import Extractor
    from ingestion.chunker import chunk_text, chunk_enrich
    from retrieval.pgvector import store_chunks
    from config.init_db import init_db
    from retrieval.kuzu_store import get_connection, init_graph_schema, upsert_triplets, link_entities_to_chunk
    from graph.entity_extraction import extract_entities

    init_db()
    # Open one Kùzu handle for the whole ingest and thread it through every
    # graph call. `graph_db` is held in a local so the Database isn't GC'd
    # while `graph_conn` is still in use.
    graph_db, graph_conn = get_connection()
    init_graph_schema(conn=graph_conn)

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

    typer.echo(f"Extracting entities (backend: {extractor or 'default'})...")
    total_triplets = 0
    for chunk in chunks:
        triplets = extract_entities(chunk.text, backend=extractor)
        upsert_triplets(triplets, conn=graph_conn)
        entity_names = list({t.source.title for t in triplets} | {t.target.title for t in triplets})
        link_entities_to_chunk(entity_names, str(chunk.chunk_id), text=chunk.text, source_id=chunk.source_id, conn=graph_conn)
        total_triplets += len(triplets)

    typer.echo(f"Done. {total_triplets} triplets added to graph.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ask a question against the ingested knowledge base."""
    from reasoning.router import route_retrieval
    from retrieval.pgvector import hybrid_search, vector_search
    from retrieval.kuzu_store import get_entity_context
    from context.builder import build_context
    from llm.generator import generate
    from embeddings.embedder import embedder

    strategy = route_retrieval(question)
    if verbose:
        typer.echo(f"Strategy: {strategy['complexity']}")

    if strategy["use_bm25"]:
        chunks = hybrid_search(question, top_k=strategy["top_k"])
    else:
        emb = embedder([question])[0]
        chunks = vector_search(emb, top_k=strategy["top_k"])

    if not chunks:
        typer.echo("No relevant context found.")
        raise typer.Exit(1)

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


if __name__ == "__main__":
    app()