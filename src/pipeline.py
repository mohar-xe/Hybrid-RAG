"""CLI entry point — ingest files and ask questions."""

import os
import uuid
from pathlib import Path
from typing import Optional

import typer

from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
app = typer.Typer(help="Hybrid-RAG CLI")


def _apply_offline_overrides(offline: bool) -> None:
    """When offline, flip all component backends to local / disable fallback."""
    if not offline:
        return
    os.environ["GENERATOR__BACKEND"] = "ollama"
    os.environ["GENERATOR__FALLBACK_ENABLED"] = "false"
    os.environ["EMBEDDING__BACKEND"] = "ollama"
    os.environ["EMBEDDING__FALLBACK_ENABLED"] = "false"
    os.environ["RERANKER__BACKEND"] = "hf"
    os.environ["RERANKER__FALLBACK_ENABLED"] = "false"
    os.environ["VERIFIER__ENABLED"] = "false"
    os.environ["EXTRACTION__BACKEND"] = "local"
    os.environ["NER__FALLBACK_ENABLED"] = "false"
    LOGGER.info("Offline mode: all components using local backends.")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File path or YouTube video ID"),
    source_type: str = typer.Option("pdf", "--type", "-t", help="pdf"),
    extractor: Optional[str] = typer.Option(
        None,
        "--extractor",
        "-e",
        help="Entity-extraction backend: 'local' (Ollama fine-tuned) or 'deepseek' "
        "(remote API). Default: EXTRACTION__BACKEND (deepseek).",
    ),
    store: bool = typer.Option(
        True,
        "--store/--no-store",
        help="Store phase: chunk + embed + write to pgvector. Disable to re-run "
        "only the graph phase against already-stored chunks (skips embedding).",
    ),
    graph: bool = typer.Option(
        True,
        "--graph/--no-graph",
        help="Graph phase: extract triplets and write to the Kùzu graph.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Use local models (Ollama/HF) instead of remote APIs.",
    ),
    embedding_backend: Optional[str] = typer.Option(
        None,
        "--embedding-backend",
        help="Embedding backend: api|ollama|sentence_transformers",
    ),
    version_label: Optional[str] = typer.Option(
        None,
        "--version-label",
        help="Explicit version label (e.g. 'v2', 'draft 3'). Overrides content-derived version.",
    ),
    supersedes: Optional[str] = typer.Option(
        None,
        "--supersedes",
        help="doc_id of the document this version replaces.",
    ),
):
    """Ingest a document into the RAG system.

    Three phases run in order: **metadata** (document-level clustering),
    **store** (chunk → embed → pgvector), and **graph** (entity/triplet → Kùzu).

    Use --no-store --graph to re-run graph extraction against already-embedded
    chunks (e.g. after a transient extraction failure).
    """
    _apply_offline_overrides(offline)
    if embedding_backend:
        os.environ["EMBEDDING__BACKEND"] = embedding_backend
    from ingestion.extractor import Extractor
    from ingestion.chunker import chunk_text, chunk_enrich
    from ingestion.document_cluster import (
        extract_document_metadata,
        create_document_cluster,
    )
    from retrieval.pgvector import store_chunks, get_chunks_by_source
    from config.init_db import init_db
    from config.settings import get_settings
    from retrieval.kuzu_store import (
        get_connection,
        init_graph_schema,
        upsert_triplets,
        link_entities_to_chunk,
    )
    from graph.entity_extraction import extract_entities_batch

    if not (store or graph):
        typer.echo("Nothing to do: both --no-store and --no-graph given.", err=True)
        raise typer.Exit(1)

    init_db()

    graph_chunks: list[tuple[str, str, str]] = []

    if store:
        ext = Extractor()
        typer.echo(f"Extracting from {path} ({source_type})...")
        if source_type == "pdf":
            text = ext.extract_pdf(path)
        else:
            typer.echo(
                f"Unsupported source type: {source_type} (only pdf is supported)",
                err=True,
            )
            raise typer.Exit(1)

        # Stage 1: Document metadata extraction (Gemini -> spaCy fallback)
        typer.echo("Extracting document metadata...")
        metadata = extract_document_metadata(text, source_id=path)
        if metadata.get("version_info") and not version_label:
            version_label = metadata["version_info"]
        typer.echo(f"  title: {metadata.get('title', 'Untitled')[:80]}")
        typer.echo(f"  type: {metadata.get('doc_type', 'N/A')}")

        is_versioned = bool(version_label or supersedes)
        doc_id = str(uuid.uuid4())
        create_document_cluster(
            doc_id,
            path,
            source_type.upper(),
            metadata,
            text,
            supersedes_doc_id=supersedes,
            is_versioned=is_versioned,
            version_label=version_label,
        )

        typer.echo("Chunking and embedding...")
        chunks = chunk_enrich(
            chunk_text(text), source_type.upper(), path, doc_id=doc_id
        )
        stored = store_chunks(chunks)
        typer.echo(f"Stored {stored} chunks in pgvector (doc_id={doc_id}).")
        graph_chunks = [(str(c.chunk_id), c.text, c.source_id) for c in chunks]

    if graph:
        if not graph_chunks:
            graph_chunks = get_chunks_by_source(path)
            if not graph_chunks:
                typer.echo(
                    f"No stored chunks found for source {path!r}. "
                    f"Run the store phase first.",
                    err=True,
                )
                raise typer.Exit(1)
            typer.echo(
                f"Loaded {len(graph_chunks)} stored chunks for graph extraction."
            )

        graph_db, graph_conn = get_connection()
        init_graph_schema(conn=graph_conn)

        typer.echo(f"Extracting entities (backend: {extractor or 'default'})...")
        s = get_settings()
        backend_name = (extractor or s.extraction.backend).lower()
        conc = (
            s.ner.concurrency
            if backend_name in ("deepseek", "api")
            else s.extraction.concurrency
        )

        total_triplets = 0
        failed = 0
        n = len(graph_chunks)
        for start in range(0, n, conc):
            batch = graph_chunks[start : start + conc]
            triplets_list = extract_entities_batch(
                [t for _, t, _ in batch], backend=extractor, max_workers=conc
            )
            for (chunk_id, text_, source_id), triplets in zip(batch, triplets_list):
                if triplets is None:
                    failed += 1
                    continue
                upsert_triplets(triplets, conn=graph_conn)
                entity_names = list(
                    {t.source.title for t in triplets}
                    | {t.target.title for t in triplets}
                )
                link_entities_to_chunk(
                    entity_names,
                    chunk_id,
                    text=text_,
                    source_id=source_id,
                    conn=graph_conn,
                )
                total_triplets += len(triplets)
            typer.echo(
                f"  processed {min(start + conc, n)}/{n} chunks (concurrency={conc})..."
            )

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
    offline: bool = typer.Option(
        False, "--offline", help="Use local models (Ollama/HF) instead of remote APIs"
    ),
    generator_backend: Optional[str] = typer.Option(
        None,
        "--generator-backend",
        help="Generator backend: api|ollama",
    ),
    embedding_backend: Optional[str] = typer.Option(
        None,
        "--embedding-backend",
        help="Embedding backend: api|ollama|sentence_transformers",
    ),
    reranker_backend: Optional[str] = typer.Option(
        None,
        "--reranker-backend",
        help="Reranker backend: api|ollama|hf",
    ),
    use_graph: Optional[bool] = typer.Option(
        None,
        "--graph/--no-graph",
        help="Override graph lookup (default: auto from complexity).",
    ),
):
    """Ask a question against the ingested knowledge base.

    Runs the full 8-stage retrieval funnel:
      0. Query interpretation (Flash-Lite or heuristic)
      1. Hard filter (doc_type, is_latest, date_after)
      2. Doc-level soft ranking (summary + question embeddings)
      3. Chunk-level hybrid search (dense + lexical)
      4. Cross-list RRF fusion
      5. Cross-encoder rerank
      6. Structural expansion (Kùzu small-to-big)
      7. Graph facts
      8. Assemble + generate
    """
    _apply_offline_overrides(offline)
    if generator_backend:
        os.environ["GENERATOR__BACKEND"] = generator_backend
    if embedding_backend:
        os.environ["EMBEDDING__BACKEND"] = embedding_backend
    if reranker_backend:
        os.environ["RERANKER__BACKEND"] = reranker_backend

    from reasoning.router import route_retrieval
    from reasoning.query_interpreter import interpret_query
    from retrieval.pgvector import (
        hard_filter_docs,
        doc_level_soft_rank,
        document_routed_search,
    )
    from retrieval.kuzu_store import get_entity_context, structural_expansion
    from retrieval.reranker import rerank
    from context.builder import build_context
    from llm.generator import generate
    from embeddings.embedder import embedder

    # Step 0: route & interpret
    strategy = route_retrieval(question)
    interpreted = interpret_query(question)
    semantic_query = interpreted.get("semantic_query", question)
    filters = interpreted.get("filters", {})
    if verbose:
        typer.echo(f"Complexity: {strategy['complexity']}")
        typer.echo(f"Filters: {filters}")

    query_emb = embedder([semantic_query])[0]

    # Step 1: hard filter (coarse SQL WHERE)
    doc_ids = hard_filter_docs(
        doc_type=filters.get("doc_type"),
        is_latest=filters.get("is_latest"),
        date_after=filters.get("date_after"),
    )

    if verbose:
        typer.echo(f"Hard filter: {len(doc_ids)} candidate docs.")

    # Step 2: doc-level soft ranking (summary + question ANN + entity boost)
    ranked_docs = doc_level_soft_rank(query_emb, semantic_query, doc_ids)
    if ranked_docs:
        doc_ids = [d[0] for d in ranked_docs]
        doc_rank_map = {d[0]: i + 1 for i, d in enumerate(ranked_docs)}
        if verbose:
            typer.echo(f"Doc-level soft rank: top {len(doc_ids)} docs.")
    else:
        doc_rank_map = None

    # Step 3-4: chunk-level hybrid search + cross-list RRF fusion
    chunks = document_routed_search(
        query_emb, question, doc_ids, doc_rank_map=doc_rank_map
    )

    if not chunks:
        typer.echo("No relevant context found.")
        raise typer.Exit(1)

    # Step 5: cross-encoder rerank
    chunks = rerank(semantic_query, chunks, top_k=top_k)
    if verbose:
        typer.echo(f"Reranked to top {len(chunks)}.")

    # Step 6: structural expansion (small-to-big)
    use_graph_flag = (
        use_graph if use_graph is not None else strategy.get("use_graph", False)
    )
    sibling_chunks: list[tuple[str, str, float]] = []
    if use_graph_flag and chunks:
        seed_ids = [c.chunk_id for c in chunks]
        sibling_chunks = structural_expansion(seed_ids)

    # Step 7: graph facts
    graph_facts = ""
    if use_graph_flag:
        from graph.entity_extraction import extract_query_entities

        entities = extract_query_entities(question)
        if entities:
            graph_facts = get_entity_context(entities)

    # Step 8: assemble + generate
    context, citations = build_context(chunks, graph_facts)
    answer = generate(question, context)

    typer.echo(f"\n{answer}\n")

    if verbose:
        typer.echo("\nSources:")
        for c in citations:
            typer.echo(f"  [{c.ref_id}] {c.source_id} — {c.text_preview}...")
        if sibling_chunks:
            typer.echo(f"\nStructural expansion: {len(sibling_chunks)} siblings.")


def _backup_graph_db() -> Optional[Path]:
    """Copy the Kùzu graph DB to a timestamped sibling backup.

    Returns the backup path, or None if there is no DB file yet. The graph is
    the product of (expensive) LLM triplet extraction during ingest, and
    `merge-graph --apply` mutates it irreversibly (duplicates are folded in and
    DETACH DELETE-d). Kùzu 0.11.x stores the DB as a single file, so a plain
    file copy is a complete, restorable snapshot — restore with:
    `cp <backup> <db_path>`.
    """
    import shutil
    from datetime import datetime
    from config.settings import get_settings

    db_path = Path(get_settings().graph.db_path)
    if not db_path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    shutil.copy2(db_path, dest)
    return dest


@app.command()
def merge_graph(
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        "-s",
        help="Cosine similarity to treat entities as near-same (default: 0.90).",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Apply the merges. Without this flag it is a dry run (list only).",
    ),
    backup: Optional[bool] = typer.Option(
        None,
        "--backup/--no-backup",
        help="With --apply: back up the graph DB before merging. If omitted you "
        "are prompted (merges are irreversible).",
    ),
):
    """Merge near-duplicate graph entities by embedding similarity.

    Run this *after* ingestion. By default it only lists the candidate pairs
    (the decision is yours); pass --apply to actually merge them. Because
    --apply is irreversible (duplicates are folded in and deleted) and the graph
    is costly to regenerate (LLM extraction), you are prompted to back up the
    graph DB first unless --backup/--no-backup is given.
    """
    from graph.merge import merge_similar_nodes

    if apply:
        do_backup = (
            backup
            if backup is not None
            else typer.confirm(
                "Merges are irreversible and the graph is LLM-generated. "
                "Back up the graph DB before applying?",
                default=True,
            )
        )
        if do_backup:
            dest = _backup_graph_db()
            typer.echo(
                f"Backed up graph DB -> {dest}"
                if dest is not None
                else "No graph DB file found yet; nothing to back up."
            )
        else:
            typer.echo("Proceeding without a backup.")

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
