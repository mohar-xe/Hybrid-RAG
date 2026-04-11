import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from config.constants import get_db_conn

LOGGER = logging.getLogger(__name__)

INSERT_CHUNK_SQL = """
INSERT INTO document_chunks
(chunk_id, chunk_index, content_text, metadata, embedding)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO NOTHING;
"""


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not JSON serializable: {value!r}") from exc


def _build_row(chunk: Mapping[str, Any]) -> tuple[Any, ...]:
    required_fields = ("chunk_id", "chunk_index", "text", "metadata")
    missing = [field for field in required_fields if field not in chunk]
    if missing:
        raise ValueError(f"Chunk missing required fields: {missing}")

    embedding = chunk.get("embeddings", chunk.get("embedding"))
    if embedding is None:
        raise ValueError("Chunk missing required field: embeddings")
    if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes, bytearray)):
        raise ValueError("Chunk field 'embeddings' must be a numeric sequence")

    return (
        chunk["chunk_id"],
        chunk["chunk_index"],
        chunk["text"],
        _to_json(chunk["metadata"]),
        list(embedding),
    )


def store_vector(chunks: Iterable[Mapping[str, Any]], conn_str: str | None = None) -> int:
    rows = [_build_row(chunk) for chunk in chunks]
    if not rows:
        LOGGER.info("No chunks provided for insertion")
        return 0

    inserted = 0
    try:
        with psycopg.connect(conn_str or get_db_conn()) as conn:
            register_vector(conn)
            with conn.transaction():
                with conn.cursor() as cur:
                    for row in rows:
                        cur.execute(INSERT_CHUNK_SQL, row)
                        inserted += cur.rowcount
    except psycopg.Error as exc:
        LOGGER.exception("Failed inserting %s chunk(s) into document_chunks", len(rows))
        raise RuntimeError("Database write failed while storing vectors") from exc

    LOGGER.info("Vector upsert complete: attempted=%s inserted=%s", len(rows), inserted)
    return inserted

def retrieve_relevant_chunks(query_embedding, limit=3):
    print("Searching database for context...")
    with psycopg.connect(get_db_conn()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content_text, metadata 
                FROM document_chunks 
                ORDER BY embedding <=> %s::vector 
                LIMIT %s;
                """,
                (query_embedding, limit)
            )
            return cur.fetchall()