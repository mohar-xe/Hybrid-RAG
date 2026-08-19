"""Deterministic ingestion of the HotpotQA corpus into the live stores.

Ingests each selected paragraph into pgvector with ``source_id = title`` and a
**deterministic** ``chunk_id`` (``f"{title}::{i}"``), so re-running is idempotent
(``store_chunks`` uses ``ON CONFLICT (chunk_id) DO NOTHING``) and reproducible.
Also creates a ``document_clusters`` entry per paragraph so the new doc-level
routing pipeline (``doc_level_soft_rank``) can score and rank paragraphs.
Optionally also extracts the knowledge graph (needed for the ``all_three`` mode
to contribute graph facts).

Batched-restructure (2026-08): API calls are grouped *across* paragraphs instead
of one request per paragraph:

* **Chunk embeddings** — ``chunk_texts_batch`` splits every paragraph first
  (pure, no API) and embeds all resulting chunk strings in ONE ``embedder()``
  pass (``ceil(1038 / BATCH_SIZE)`` requests instead of ~997).
* **Summary embeddings** — all summaries embedded in one batched pass and
  passed via ``create_document_cluster(summary_embedding=...)`` (~8 requests
  instead of ~997).
* **KG triplet extraction** — a single ``extract_entities_batch`` call over ALL
  chunk texts; ``_extract_entities_gemini_batch`` internally groups them at
  ``NER__BATCH_SIZE`` (50) per Gemini call. 1038 chunks -> ~21 Gemini calls
  instead of ~997 (one per paragraph), which keeps free-tier quotas (250 RPD)
  and cuts the 5-RPM wall-clock from ~3.3 h to ~5 min.

Results stay order-preserving at every step, so per-paragraph writes and the
Kùzu link loop are unchanged in behaviour.

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

    Creates:
      * Chunks with deterministic IDs and ``doc_id = title``.
      * A ``document_clusters`` row per paragraph for doc-level routing.
      * (optional) KG triplets in Kùzu.

    Returns counts: ``{"titles", "chunks", "graph_triplets", "graph_failed"}``.
    """
    import uuid as _uuid

    from config.init_db import init_db
    from ingestion.chunker import chunk_enrich, chunk_texts_batch
    from ingestion.document_cluster import create_document_cluster
    from retrieval.pgvector import store_chunks
    from constants.logger import setup_logger

    logger = setup_logger("evaluation.corpus")
    init_db()  # idempotent: ensures schema + indexes exist

    graph_conn = None
    if with_graph:
        from retrieval.kuzu_store import get_connection, init_graph_schema

        _db, graph_conn = get_connection()
        init_graph_schema(conn=graph_conn)

    titles = sorted(corpus)  # deterministic ingestion order
    texts = [_CONTROL.sub("", corpus[t]).strip() for t in titles]

    # 1. Split every paragraph (pure) and embed ALL chunks in one batched pass.
    drafts_per_title = chunk_texts_batch(texts)

    # 2. Embed ALL summaries in one batched pass (shared with the cluster rows).
    from embeddings.embedder import embedder

    summaries = [t[:500] for t in texts]
    summary_embeddings = embedder(summaries)

    # 3. KG extraction: ONE batched call over every chunk text. The Gemini
    #    backend groups them internally at NER__BATCH_SIZE (50) per request and
    #    returns results in input order — zipped back to chunks below.
    triplets_per_chunk: list[list | None] | None = None
    if with_graph:
        from graph.entity_extraction import extract_entities_batch

        all_chunk_texts = [
            d.text for drafts in drafts_per_title for d in drafts
        ]
        triplets_per_chunk = extract_entities_batch(
            all_chunk_texts, backend=backend
        )

    # 4. Build all chunks (deterministic ids, doc_id wiring) and store once.
    all_chunks: list = []
    chunk_triplet_pairs: list[tuple] = []
    n_triplets = n_failed = 0
    n_chunks = 0
    n_titles = 0
    chunk_text_idx = 0  # position into triplets_per_chunk (order-preserving)

    for idx, (title, text) in enumerate(zip(titles, texts), start=1):
        if not text:
            continue
        n_titles += 1

        doc_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, title))
        chunks = chunk_enrich(drafts_per_title[idx - 1], "PDF", title, doc_id=doc_id)
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{title}::{i}"
        all_chunks.extend(chunks)
        n_chunks += len(chunks)

        create_document_cluster(
            doc_id=doc_id,
            source_id=title,
            source_type="hotpotqa",
            metadata={
                "title": title,
                "summary": summaries[idx - 1],
                "synthetic_questions": [],
                "doc_type": "",
                "topic_tags": [],
                "entities": [],
                "content_date": None,
                "version_info": None,
            },
            text=text,
            summary_embedding=summary_embeddings[idx - 1],
        )

        if with_graph and triplets_per_chunk is not None:
            for chunk in chunks:
                chunk_triplet_pairs.append(
                    (chunk, triplets_per_chunk[chunk_text_idx])
                )
                chunk_text_idx += 1

        if idx % progress_every == 0:
            logger.info(
                f"Ingested {idx}/{len(titles)} paragraphs "
                f"({n_chunks} chunks)."
            )

    store_chunks(all_chunks)

    if with_graph:
        n_triplets, n_failed = _write_graph(chunk_triplet_pairs, graph_conn)

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


def _write_graph(pairs: list[tuple], conn) -> tuple[int, int]:
    """Write ``(chunk, triplets|None)`` pairs to Kùzu. Returns (triplets, failures)."""
    from retrieval.kuzu_store import upsert_triplets, link_entities_to_chunk

    n_triplets = n_failed = 0
    for chunk, triplets in pairs:
        if triplets is None:
            n_failed += 1
            continue
        upsert_triplets(triplets, conn=conn)
        entity_names = list(
            {t.source.title for t in triplets} | {t.target.title for t in triplets}
        )
        link_entities_to_chunk(
            entity_names,
            str(chunk.chunk_id),
            text=chunk.text,
            source_id=chunk.source_id,
            conn=conn,
        )
        n_triplets += len(triplets)
    return n_triplets, n_failed