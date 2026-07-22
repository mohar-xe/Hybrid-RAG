"""Vector retrieval with hybrid cosine + BM25 scoring, plus document-level routing."""

from dataclasses import dataclass

import re

import psycopg

from config.settings import get_settings
from embeddings.embedder import embedder
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_id: str
    score: float
    source_type: str = ""
    chunk_index: int = 0
    doc_id: str | None = None


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


_SELECT_COLS = """SELECT chunk_id, text, source_id, source_type, chunk_index,
                         1 - (embedding <=> %s::vector) AS score"""

_SELECT_COLS_DOC = """SELECT chunk_id, text, source_id, source_type, chunk_index,
                         1 - (embedding <=> %s::vector) AS score, doc_id"""


def _rows_to_chunks(rows) -> list["RetrievedChunk"]:
    return [
        RetrievedChunk(
            chunk_id=r[0],
            text=r[1],
            source_id=r[2],
            source_type=r[3],
            chunk_index=r[4],
            score=r[5],
        )
        for r in rows
    ]


def _rows_to_chunks_doc(rows) -> list["RetrievedChunk"]:
    return [
        RetrievedChunk(
            chunk_id=r[0],
            text=r[1],
            source_id=r[2],
            source_type=r[3],
            chunk_index=r[4],
            score=r[5],
            doc_id=r[6],
        )
        for r in rows
    ]


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


# ---------------------------------------------------------------------------
# Document-level soft ranking (Stage 2)
# ---------------------------------------------------------------------------


def _extract_query_entities(text: str) -> set[str]:
    """Simple heuristic entity extraction: capitalized words, not stop words."""
    stop = {
        "what",
        "who",
        "why",
        "when",
        "where",
        "how",
        "which",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "may",
        "might",
        "shall",
        "should",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "and",
        "or",
        "but",
        "not",
        "no",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "into",
        "through",
        "about",
    }
    entities: set[str] = set()
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        token = match.group().strip()
        if token.lower() not in stop and not any(c.isdigit() for c in token):
            entities.add(token)
    return entities


def doc_level_soft_rank(
    query_embedding: list[float],
    query_text: str,
    doc_ids: list[str] | None = None,
    *,
    summary_k: int = 50,
    question_k: int = 50,
    top_k: int = 25,
) -> list[tuple[str, float]]:
    """Stage 2: score candidate docs via summary/question ANN + entity boost.

    Returns top-k ``(doc_id, rrf_score)`` pairs ordered descending by RRF score.
    The caller should pass only these doc_ids (and their rank map) to Stage 3.
    Returns an empty list when no docs are available.
    """
    q = _vec_literal(query_embedding)

    # --- Summary embedding ANN ---
    summary_scores: dict[str, float] = {}
    if doc_ids:
        sql = """
            SELECT doc_id, 1 - (summary_embedding <=> %s::vector) AS sim
            FROM document_clusters
            WHERE doc_id = ANY(%s) AND summary_embedding IS NOT NULL
            ORDER BY summary_embedding <=> %s::vector
            LIMIT %s
        """
        params = (q, doc_ids, q, summary_k)
    else:
        sql = """
            SELECT doc_id, 1 - (summary_embedding <=> %s::vector) AS sim
            FROM document_clusters
            WHERE summary_embedding IS NOT NULL
            ORDER BY summary_embedding <=> %s::vector
            LIMIT %s
        """
        params = (q, q, summary_k)

    with psycopg.connect(settings.database.conninfo) as conn:
        rows = conn.execute(sql, params).fetchall()
        for doc_id, sim in rows:
            summary_scores[doc_id] = sim

    # --- Question embedding ANN ---
    question_scores: dict[str, list[float]] = {}
    if doc_ids:
        sql = """
            SELECT dq.doc_id, 1 - (dq.embedding <=> %s::vector) AS sim
            FROM document_questions dq
            WHERE dq.doc_id = ANY(%s)
            ORDER BY dq.embedding <=> %s::vector
            LIMIT %s
        """
        params = (q, doc_ids, q, question_k)
    else:
        sql = """
            SELECT dq.doc_id, 1 - (dq.embedding <=> %s::vector) AS sim
            FROM document_questions dq
            ORDER BY dq.embedding <=> %s::vector
            LIMIT %s
        """
        params = (q, q, question_k)

    with psycopg.connect(settings.database.conninfo) as conn:
        rows = conn.execute(sql, params).fetchall()
        for doc_id, sim in rows:
            question_scores.setdefault(doc_id, []).append(sim)

    question_max: dict[str, float] = {
        d: max(sims) for d, sims in question_scores.items()
    }

    # --- Entity/topic overlap boost ---
    query_entities = _extract_query_entities(query_text)
    lexical_scores: dict[str, int] = {}
    candidate_ids = doc_ids or list(summary_scores.keys()) or list(question_max.keys())
    if query_entities and candidate_ids:
        with psycopg.connect(settings.database.conninfo) as conn:
            rows = conn.execute(
                """SELECT doc_id, topic_tags, entities
                   FROM document_clusters
                   WHERE doc_id = ANY(%s)""",
                (candidate_ids,),
            ).fetchall()
        for doc_id, tags, entities_json in rows:
            doc_terms: set[str] = set(tags or [])
            if entities_json:
                for ent in entities_json:
                    name = ent.get("name", "")
                    if name:
                        doc_terms.add(name)
            overlap = len(query_entities & doc_terms)
            if overlap:
                lexical_scores[doc_id] = overlap

    # --- RRF fusion ---
    scored_docs: dict[str, float] = {}
    for doc_id in candidate_ids:
        rrf_total = 0.0

        rank_summary = summary_scores.get(doc_id)
        if rank_summary is not None:
            sorted_summary = sorted(
                summary_scores.items(), key=lambda x: x[1], reverse=True
            )
            for i, (sid, _) in enumerate(sorted_summary, start=1):
                if sid == doc_id:
                    rrf_total += _rrf_score(i)
                    break

        rank_question = question_max.get(doc_id)
        if rank_question is not None:
            sorted_q = sorted(question_max.items(), key=lambda x: x[1], reverse=True)
            for i, (qid, _) in enumerate(sorted_q, start=1):
                if qid == doc_id:
                    rrf_total += _rrf_score(i)
                    break

        lex_rank = lexical_scores.get(doc_id)
        if lex_rank is not None:
            sorted_lex = sorted(
                lexical_scores.items(), key=lambda x: x[1], reverse=True
            )
            for i, (lid, _) in enumerate(sorted_lex, start=1):
                if lid == doc_id:
                    rrf_total += _rrf_score(i)
                    break

        if rrf_total > 0:
            scored_docs[doc_id] = rrf_total

    if not scored_docs:
        # If every doc scored 0 (no signals at all), return all candidate_ids
        # with a neutral score so the pipeline has something to work with.
        return [(d, 1.0) for d in candidate_ids[:top_k]]

    ranked = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    LOGGER.info(
        "Doc-level soft rank: %d candidates -> %d scored -> top %d.",
        len(candidate_ids),
        len(scored_docs),
        len(ranked),
    )
    return ranked


# ---------------------------------------------------------------------------
# Document-level routed chunk search (Stage 3 + 4)
# ---------------------------------------------------------------------------


def _chunk_dense_search(
    query_embedding: list[float],
    doc_ids: list[str] | None = None,
    top_k: int = 50,
) -> list[RetrievedChunk]:
    """Dense ANN over chunks, optionally restricted to document IDs."""
    q = _vec_literal(query_embedding)
    if doc_ids:
        sql = (
            _SELECT_COLS_DOC
            + """
              FROM chunks
              WHERE doc_id = ANY(%s)
              ORDER BY embedding <=> %s::vector
              LIMIT %s"""
        )
        params = (q, doc_ids, q, top_k)
    else:
        sql = (
            _SELECT_COLS_DOC
            + """
              FROM chunks
              ORDER BY embedding <=> %s::vector
              LIMIT %s"""
        )
        params = (q, q, top_k)

    with psycopg.connect(settings.database.conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return _rows_to_chunks_doc(cur.fetchall())


def _chunk_lexical_search(
    query: str,
    doc_ids: list[str] | None = None,
    top_k: int = 50,
) -> list[RetrievedChunk]:
    """Lexical (full-text) search over chunks, optionally restricted to document IDs."""
    if doc_ids:
        sql = """
              SELECT chunk_id, text, source_id, source_type, chunk_index,
                     ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score,
                     doc_id
              FROM chunks
              WHERE tsv @@ plainto_tsquery('english', %s)
                AND doc_id = ANY(%s)
              ORDER BY score DESC
              LIMIT %s"""
        params = (query, query, doc_ids, top_k)
    else:
        sql = """
              SELECT chunk_id, text, source_id, source_type, chunk_index,
                     ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score,
                     doc_id
              FROM chunks
              WHERE tsv @@ plainto_tsquery('english', %s)
              ORDER BY score DESC
              LIMIT %s"""
        params = (query, query, top_k)

    with psycopg.connect(settings.database.conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return _rows_to_chunks_doc(cur.fetchall())


def document_routed_search(
    query_embedding: list[float],
    query_text: str,
    doc_ids: list[str] | None = None,
    doc_rank_map: dict[str, int] | None = None,
    *,
    dense_k: int = 50,
    lexical_k: int = 50,
    fusion_top_k: int = 75,
) -> list[RetrievedChunk]:
    """Document-routed chunk search (Stage 3 + 4 of the 8-stage funnel).

    Args:
        query_embedding: embedded semantic_query (from query interpreter).
        query_text: raw query text for lexical search.
        doc_ids: pre-filtered doc IDs from Stage 2 (top-ranked docs).
        doc_rank_map: doc_id -> 1-based rank from Stage 2 doc-level ranking,
                      used for cross-list RRF fusion (Stage 4).
        dense_k: per-retriever candidate count.
        lexical_k: per-retriever candidate count.
        fusion_top_k: candidates passed to the reranker.

    Returns RRF-fused dense + lexical + doc-level results ordered by fused score.
    """
    dense = _chunk_dense_search(query_embedding, doc_ids, top_k=dense_k)
    lexical = _chunk_lexical_search(query_text, doc_ids, top_k=lexical_k)

    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    # Signal A (doc-level) + B (dense chunk)
    for rank, c in enumerate(dense, start=1):
        a_score = _rrf_score(doc_rank_map.get(c.doc_id, 1)) if doc_rank_map else 0
        scores[c.chunk_id] = a_score + _rrf_score(rank)  # A + B
        chunk_map[c.chunk_id] = c

    # Signal C (lexical chunk) — merge with existing A+B if present
    for rank, c in enumerate(lexical, start=1):
        c_score = _rrf_score(rank)  # C
        if c.chunk_id in scores:
            scores[c.chunk_id] += c_score  # was A+B, now A+B+C
        else:
            a_score = _rrf_score(doc_rank_map.get(c.doc_id, 1)) if doc_rank_map else 0
            scores[c.chunk_id] = a_score + c_score  # A + C
            chunk_map[c.chunk_id] = c

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:fusion_top_k]
    results = []
    for chunk_id, score in ranked:
        c = chunk_map[chunk_id]
        c.score = score
        results.append(c)

    LOGGER.info(
        "Document-routed search: %d dense + %d lexical + %d doc-signal -> %d fused candidates.",
        len(dense),
        len(lexical),
        len(doc_rank_map) if doc_rank_map else 0,
        len(results),
    )
    return results


def hard_filter_docs(
    doc_type: str | None = None,
    is_latest: bool | None = None,
    date_after: str | None = None,
    entity_filter: list[str] | None = None,
) -> list[str]:
    """Hard SQL WHERE filter over document_clusters (Stage 1).

    Returns matching ``doc_id`` values. Safety valve: retries by dropping
    filters one at a time so a parser mistake never produces empty results.
    """
    clauses: list[str] = []
    params: list = []

    if doc_type:
        clauses.append("doc_type = %s")
        params.append(doc_type)
    if is_latest:
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM document_clusters AS d2 WHERE d2.supersedes_doc_id = d.doc_id)"
        )
    if date_after:
        clauses.append("content_date >= %s::date")
        params.append(date_after)

    base = "SELECT doc_id FROM document_clusters d"
    where = " WHERE " + " AND ".join(clauses) if clauses else ""

    for attempt in range(4):
        sql = base + where
        with psycopg.connect(settings.database.conninfo) as conn:
            rows = conn.execute(sql, params).fetchall()
        if rows:
            return [r[0] for r in rows]

        if attempt == 0 and is_latest:
            LOGGER.info("Hard filter returned 0 rows; retrying without is_latest.")
            where = _drop_clause(where, "NOT EXISTS")
            is_latest = None
        elif attempt == 1 and doc_type:
            LOGGER.info("Hard filter returned 0 rows; retrying without doc_type.")
            where = _drop_clause(where, "doc_type")
            doc_type = None
        elif attempt == 2 and date_after:
            LOGGER.info("Hard filter returned 0 rows; retrying without date_after.")
            where = _drop_clause(where, "content_date")
            date_after = None
        else:
            break

    LOGGER.info("Hard filter returned all docs (safety valve).")
    with psycopg.connect(settings.database.conninfo) as conn:
        rows = conn.execute("SELECT doc_id FROM document_clusters").fetchall()
    return [r[0] for r in rows]


def _drop_clause(where: str, keyword: str) -> str:
    """Remove a SQL clause by keyword from a WHERE fragment."""
    parts = [p.strip() for p in where.replace("WHERE ", "").split("AND")]
    parts = [p for p in parts if keyword.lower() not in p.lower()]
    if parts:
        return " WHERE " + " AND ".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Legacy: cluster-routed search (kept for backward compatibility)
# ---------------------------------------------------------------------------


def cluster_routed_search(
    query_embedding: list[float],
    *,
    coarse_k: int = 10,
    n_clusters: int = 5,
    fine_k: int = 15,
    global_k: int = 5,
    score_gate: float = 0.35,
) -> list["RetrievedChunk"]:
    """Coarse->fine cluster-routed dense retrieval (legacy).

    Replaced by ``document_routed_search``. Kept for backward compatibility
    until the eval harness and all callers migrate.
    """
    q = _vec_literal(query_embedding)

    with psycopg.connect(settings.database.conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cluster_id, 1 - (embedding <=> %s::vector) AS sim
                FROM chunks
                WHERE is_medoid AND cluster_id IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q, q, coarse_k),
            )
            coarse = cur.fetchall()

            if not coarse or max(r[1] for r in coarse) < score_gate:
                reason = (
                    "no medoids (run reindex)" if not coarse else "below score gate"
                )
                LOGGER.info(
                    f"Coarse routing skipped ({reason}); flat global retrieval."
                )
                cur.execute(
                    _SELECT_COLS
                    + """
                    FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
                    """,
                    (q, q, fine_k + global_k),
                )
                return _rows_to_chunks(cur.fetchall())

            top_clusters: list[int] = []
            for cid, _sim in coarse:
                if cid not in top_clusters:
                    top_clusters.append(cid)
                if len(top_clusters) >= n_clusters:
                    break

            cur.execute(
                _SELECT_COLS
                + """
                FROM chunks
                WHERE cluster_id = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q, top_clusters, q, fine_k),
            )
            fine = _rows_to_chunks(cur.fetchall())

            cur.execute(
                _SELECT_COLS
                + """
                FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (q, q, global_k),
            )
            glob = _rows_to_chunks(cur.fetchall())

    by_id: dict[str, RetrievedChunk] = {}
    for c in fine + glob:
        if c.chunk_id not in by_id or c.score > by_id[c.chunk_id].score:
            by_id[c.chunk_id] = c
    results = sorted(by_id.values(), key=lambda c: c.score, reverse=True)
    LOGGER.info(
        f"Cluster-routed retrieval: {len(top_clusters)} clusters, "
        f"{len(fine)} fine + {len(glob)} global -> {len(results)} candidates."
    )
    return results


# ---------------------------------------------------------------------------
# Shared helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def vector_search(
    query_embedding: list[float], top_k: int = 20
) -> list[RetrievedChunk]:
    emb_literal = _vec_literal(query_embedding)

    sql = """
        SELECT chunk_id, text, source_id, source_type, chunk_index,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with psycopg.connect(settings.database.conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (emb_literal, emb_literal, top_k))
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            chunk_id=r[0],
            text=r[1],
            source_id=r[2],
            source_type=r[3],
            chunk_index=r[4],
            score=r[5],
        )
        for r in rows
    ]


def bm25_search(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    sql = """
        SELECT chunk_id, text, source_id, source_type, chunk_index,
               ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
        FROM chunks
        WHERE tsv @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
    """
    with psycopg.connect(settings.database.conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, query, top_k))
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            chunk_id=r[0],
            text=r[1],
            source_id=r[2],
            source_type=r[3],
            chunk_index=r[4],
            score=r[5],
        )
        for r in rows
    ]


def hybrid_search(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    query_emb = embedder([query])[0]

    vec_results = vector_search(query_emb, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)

    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(vec_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + _rrf_score(rank)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + _rrf_score(rank)
        chunk_map[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for chunk_id, score in ranked:
        c = chunk_map[chunk_id]
        c.score = score
        results.append(c)

    LOGGER.info(f"Hybrid search returned {len(results)} chunks")
    return results


def store_chunks(chunks: list, conn_info: str | None = None) -> int:
    conninfo = conn_info or settings.database.conninfo

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            for c in chunks:
                emb_str = _vec_literal(c.embeddings)
                if c.doc_id:
                    cur.execute(
                        """INSERT INTO chunks (chunk_id, text, embedding, source_type, source_id, chunk_index, keyword, doc_id)
                           VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s)
                           ON CONFLICT (chunk_id) DO NOTHING""",
                        (
                            c.chunk_id,
                            c.text,
                            emb_str,
                            c.source_type,
                            c.source_id,
                            c.chunk_index,
                            c.keyword,
                            c.doc_id,
                        ),
                    )
                else:
                    cur.execute(
                        """INSERT INTO chunks (chunk_id, text, embedding, source_type, source_id, chunk_index, keyword)
                           VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
                           ON CONFLICT (chunk_id) DO NOTHING""",
                        (
                            c.chunk_id,
                            c.text,
                            emb_str,
                            c.source_type,
                            c.source_id,
                            c.chunk_index,
                            c.keyword,
                        ),
                    )
        conn.commit()

    LOGGER.info(f"Stored {len(chunks)} chunks.")
    return len(chunks)


def get_chunks_by_source(
    source_id: str, conn_info: str | None = None
) -> list[tuple[str, str, str]]:
    """Return ``(chunk_id, text, source_id)`` for all stored chunks of a source.

    Lets the graph-extraction phase run against already-embedded chunks without
    re-chunking/re-embedding (see ``pipeline ingest --no-store --graph``).
    """
    conninfo = conn_info or settings.database.conninfo
    with psycopg.connect(conninfo) as conn:
        rows = conn.execute(
            "SELECT chunk_id, text, source_id FROM chunks WHERE source_id = %s ORDER BY chunk_index",
            (source_id,),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]
