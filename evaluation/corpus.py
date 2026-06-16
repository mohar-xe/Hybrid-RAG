"""Deterministic ingestion of the HotpotQA corpus into the live stores.

Ingests each selected paragraph into pgvector with ``source_id = title`` and a
**deterministic** ``chunk_id`` (``f"{title}::{i}"``), so re-running is idempotent
(``store_chunks`` uses ``ON CONFLICT (chunk_id) DO NOTHING``) and reproducible.
Optionally also extracts the knowledge graph (needed for the ``all_three`` mode
to contribute graph facts).

All heavy ``src`` imports are function-local so importing this module is cheap
and does not require a database/Ollama to be running.
"""

from __future__ import annotations

import re

# Postgres `text` rejects NUL; HotpotQA is clean but guard defensively.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def ingest_corpus(
    corpus: dict[str, str],
    *,
    with_graph: bool = False,
    backend: str | None = None,
    progress_every: int = 50,
) -> dict[str, int]:
    """Ingest ``{title: text}`` into pgvector (and optionally the Kùzu graph).

    Returns counts: ``{"titles", "chunks", "graph_triplets", "graph_failed"}``.
    """
    from config.init_db import init_db
    from ingestion.chunker import chunk_text, chunk_enrich
    from retrieval.pgvector import store_chunks
    from constants.logger import setup_logger

    logger = setup_logger("evaluation.corpus")
    init_db()  # idempotent: ensures schema + indexes exist

    graph_conn = None
    if with_graph:
        from retrieval.kuzu_store import get_connection, init_graph_schema

        _db, graph_conn = get_connection()
        init_graph_schema(conn=graph_conn)

    n_titles = n_chunks = n_triplets = n_failed = 0
    titles = sorted(corpus)  # deterministic ingestion order

    for idx, title in enumerate(titles, start=1):
        text = _CONTROL.sub("", corpus[title]).strip()
        if not text:
            continue

        chunks = chunk_enrich(chunk_text(text), "PDF", title)
        # Deterministic, idempotent chunk ids (override the random uuid4).
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{title}::{i}"
        store_chunks(chunks)
        n_titles += 1
        n_chunks += len(chunks)

        if with_graph:
            n_t, n_f = _ingest_graph(chunks, backend, graph_conn)
            n_triplets += n_t
            n_failed += n_f

        if idx % progress_every == 0:
            logger.info(f"Ingested {idx}/{len(titles)} paragraphs ({n_chunks} chunks).")

    logger.info(
        f"Corpus ingest done: {n_titles} paragraphs, {n_chunks} chunks, "
        f"{n_triplets} triplets ({n_failed} extraction failures)."
    )
    return {
        "titles": n_titles,
        "chunks": n_chunks,
        "graph_triplets": n_triplets,
        "graph_failed": n_failed,
    }


def _ingest_graph(chunks: list, backend: str | None, conn) -> tuple[int, int]:
    """Extract triplets for ``chunks`` and write them to Kùzu. Returns (triplets, failures)."""
    from graph.entity_extraction import extract_entities_batch
    from retrieval.kuzu_store import upsert_triplets, link_entities_to_chunk

    triplets_list = extract_entities_batch([c.text for c in chunks], backend=backend)
    n_triplets = n_failed = 0
    for chunk, triplets in zip(chunks, triplets_list):
        if triplets is None:
            n_failed += 1
            continue
        upsert_triplets(triplets, conn=conn)
        entity_names = list({t.source.title for t in triplets} | {t.target.title for t in triplets})
        link_entities_to_chunk(
            entity_names, str(chunk.chunk_id), text=chunk.text,
            source_id=chunk.source_id, conn=conn,
        )
        n_triplets += len(triplets)
    return n_triplets, n_failed
