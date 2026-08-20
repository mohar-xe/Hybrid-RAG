"""Ingestion-state check for fast_eval.

The paid eval's first step is: **has the corpus already been ingested?** If yes,
skip ingestion and go straight to retrieval. This module answers that with cheap
count queries against the live stores (pgvector + Kùzu), and can run ingestion
when they come back empty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestState:
    vector_ready: bool = False
    graph_ready: bool = False

    @property
    def ready(self) -> bool:
        return self.vector_ready


def check_ingested(source_type: str = "hotpotqa") -> IngestState:
    """Return what is already ingested (vector + graph) by counting store rows.

    Cheap and side-effect free: a couple of COUNT queries, no writes.
    """
    vector_ready = _vector_count(source_type) > 0
    graph_ready = _graph_count() > 0
    return IngestState(vector_ready=vector_ready, graph_ready=graph_ready)


def _vector_count(source_type: str) -> int:
    try:
        import psycopg
        from config.settings import get_settings
        from config.init_db import init_db

        init_db()
        settings = get_settings()
        with psycopg.connect(settings.database.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM document_clusters WHERE source_type = %s",
                    (source_type,),
                )
                return int(cur.fetchone()[0])
    except Exception as exc:  # pragma: no cover - depends on live DB
        import logging

        logging.getLogger("fast_eval.check_ingest").warning(
            "Vector-count check failed (%s); treating as not ingested.", exc
        )
        return 0


def _graph_count() -> int:
    try:
        import logging
        from retrieval.kuzu_store import get_connection

        _db, conn = get_connection()
        try:
            rows = conn.execute("MATCH (c:Chunk) RETURN count(c) AS n")
            row = rows.get_next()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - depends on live Kùzu
        import logging

        logging.getLogger("fast_eval.check_ingest").warning(
            "Graph-count check failed (%s); treating as not ingested.", exc
        )
        return 0


def ingest_if_missing(
    corpus: dict[str, str], *, with_graph: bool, source_type: str = "hotpotqa"
) -> IngestState:
    """Ingest the corpus when the check says it is missing; never re-ingests.

    Returns the final state after any ingestion. By default does not re-ingest an
    already-present corpus (idempotency also live in ``store_chunks``).
    """
    state = check_ingested(source_type)
    if state.vector_ready:
        return state

    from evaluation.corpus import ingest_corpus

    ingest_corpus(corpus, with_graph=with_graph)
    return check_ingested(source_type)
