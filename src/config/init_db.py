import logging
import psycopg

from config.constants import settings

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CREATE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS hybrid_rag_chunks (
    chunk_id VARCHAR PRIMARY KEY,
    chunk_index INTEGER NOT NULL,
    content_text TEXT NOT NULL,
    metadata JSONB,
    embedding vector(384));
    
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
ON hybrid_rag_chunks USING hnsw (embedding vector_cosine_ops);"""

def init_postgres_schema(conn_str: str | None = None):
    with psycopg.connect(conn_str or settings.database_url) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                try:
                    cur.execute(CREATE_SCHEMA_SQL)
                    LOGGER.info("Postgres pgvector schema initialized successfully.")
                except Exception as e:
                    LOGGER.error(f"Failed to initialize schema: {e}")