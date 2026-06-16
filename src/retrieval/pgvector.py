"""Vector retrieval with hybrid cosine + BM25 scoring."""

from dataclasses import dataclass

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


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


_SELECT_COLS = """SELECT chunk_id, text, source_id, source_type, chunk_index,
                         1 - (embedding <=> %s::vector) AS score"""


def _rows_to_chunks(rows) -> list["RetrievedChunk"]:
    return [
        RetrievedChunk(
            chunk_id=r[0], text=r[1], source_id=r[2],
            source_type=r[3], chunk_index=r[4], score=r[5],
        )
        for r in rows
    ]


def cluster_routed_search(
    query_embedding: list[float],
    *,
    coarse_k: int = 10,
    n_clusters: int = 5,
    fine_k: int = 15,
    global_k: int = 5,
    score_gate: float = 0.35,
) -> list["RetrievedChunk"]:
    """Coarse->fine cluster-routed dense retrieval (Stage 3).

    1. Coarse: ANN over medoids only -> top ``coarse_k``.
    2. Gate: if the best medoid similarity < ``score_gate`` (or no medoids exist
       yet, i.e. ``reindex`` hasn't run), fall back to a flat global ANN.
    3. Pick the top ``n_clusters`` distinct clusters by medoid similarity.
    4. Fine: cluster-filtered ANN (``cluster_id = ANY(top)``) -> top ``fine_k``.
    5. Global fallback: a parallel unfiltered ANN -> top ``global_k``, merged and
       de-duplicated with the fine results (keeping the higher score).

    Returns up to ``fine_k + global_k`` candidates ordered by similarity, ready
    for the reranker to trim to the final top-k.
    """
    q = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with psycopg.connect(settings.database.conninfo) as conn:
        with conn.cursor() as cur:
            # 1. Coarse pass over medoids only.
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

            # 2. Gate -> global retrieval if weak or unindexed.
            if not coarse or max(r[1] for r in coarse) < score_gate:
                reason = "no medoids (run reindex)" if not coarse else "below score gate"
                LOGGER.info(f"Coarse routing skipped ({reason}); flat global retrieval.")
                cur.execute(
                    _SELECT_COLS + """
                    FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
                    """,
                    (q, q, fine_k + global_k),
                )
                return _rows_to_chunks(cur.fetchall())

            # 3. Top distinct clusters by medoid similarity.
            top_clusters: list[int] = []
            for cid, _sim in coarse:
                if cid not in top_clusters:
                    top_clusters.append(cid)
                if len(top_clusters) >= n_clusters:
                    break

            # 4. Fine, cluster-filtered ANN.
            cur.execute(
                _SELECT_COLS + """
                FROM chunks
                WHERE cluster_id = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q, top_clusters, q, fine_k),
            )
            fine = _rows_to_chunks(cur.fetchall())

            # 5. Parallel global fallback, merged + de-duplicated.
            cur.execute(
                _SELECT_COLS + """
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


def vector_search(query_embedding: list[float], top_k: int = 20) -> list[RetrievedChunk]:
    emb_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

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
            chunk_id=r[0], text=r[1], source_id=r[2],
            source_type=r[3], chunk_index=r[4], score=r[5],
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
            chunk_id=r[0], text=r[1], source_id=r[2],
            source_type=r[3], chunk_index=r[4], score=r[5],
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
                emb_str = "[" + ",".join(str(x) for x in c.embeddings) + "]"
                cur.execute(
                    """INSERT INTO chunks (chunk_id, text, embedding, source_type, source_id, chunk_index, keyword)
                       VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
                       ON CONFLICT (chunk_id) DO NOTHING""",
                    (c.chunk_id, c.text, emb_str, c.source_type, c.source_id, c.chunk_index, c.keyword),
                )
        conn.commit()

    LOGGER.info(f"Stored {len(chunks)} chunks.")
    return len(chunks)


def get_chunks_by_source(source_id: str, conn_info: str | None = None) -> list[tuple[str, str, str]]:
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