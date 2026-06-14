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