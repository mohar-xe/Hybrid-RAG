import psycopg

from constants.logger import setup_logger
from constants.exceptions import DatabaseError
from config.settings import get_settings

LOGGER = setup_logger(__name__)
settings = get_settings()

ENABLE_EXTENSION = """CREATE EXTENSION IF NOT EXISTS vector"""

SQL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks(
chunk_id TEXT PRIMARY KEY,
text TEXT NOT NULL,
embedding vector(256),
source_type TEXT NOT NULL,
source_id TEXT NOT NULL,
chunk_index INTEGER NOT NULL,
keyword TEXT[],
doc_id UUID,
tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
)"""

SQL_DOCUMENT_CLUSTERS = """
CREATE TABLE IF NOT EXISTS document_clusters(
doc_id UUID PRIMARY KEY,
source_id TEXT NOT NULL,
source_type TEXT NOT NULL,
title TEXT NOT NULL,
summary TEXT,
summary_embedding vector(256),
doc_type TEXT,
topic_tags TEXT[],
content_date DATE,
supersedes_doc_id UUID REFERENCES document_clusters(doc_id),
is_versioned BOOLEAN NOT NULL DEFAULT FALSE,
version_label TEXT,
entities JSONB,
metadata_json JSONB,
ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)"""

SQL_DOCUMENT_QUESTIONS = """
CREATE TABLE IF NOT EXISTS document_questions(
id UUID PRIMARY KEY,
doc_id UUID NOT NULL REFERENCES document_clusters(doc_id) ON DELETE CASCADE,
question TEXT NOT NULL,
embedding vector(256) NOT NULL
)"""

CREATE_INDEX_EMBEDDING = """CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"""
CREATE_INDEX_TSVECTOR = (
    """CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv)"""
)
CREATE_INDEX_DOC_CLUSTER_SOURCE = """CREATE INDEX IF NOT EXISTS idx_doc_clusters_source ON document_clusters (source_id)"""
CREATE_INDEX_DOC_CLUSTER_TYPE = """CREATE INDEX IF NOT EXISTS idx_doc_clusters_type ON document_clusters (doc_type)"""
CREATE_INDEX_DOC_CLUSTER_SUPERSEDES = """CREATE INDEX IF NOT EXISTS idx_doc_clusters_supersedes ON document_clusters (supersedes_doc_id)"""
CREATE_INDEX_DOC_QUESTIONS_DOC = """CREATE INDEX IF NOT EXISTS idx_doc_questions_doc ON document_questions (doc_id)"""
CREATE_INDEX_DOC_QUESTIONS_EMBEDDING = """CREATE INDEX IF NOT EXISTS idx_doc_questions_embedding ON document_questions USING hnsw (embedding vector_cosine_ops)"""
CREATE_INDEX_CHUNKS_DOC = (
    """CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id)"""
)

# Idempotent migrations for columns added after the initial schema.
# cluster_id/is_medoid are no longer created (old K-Means/medoid path removed);
# existing columns on old data remain as dead schema.
MIGRATIONS = [
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS cluster_id INTEGER",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_medoid BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_id UUID",
]


def init_db():
    if not settings.database.init_schema:
        LOGGER.info(
            "DATABASE__INIT_SCHEMA is false — skipping schema init (read-only mode)."
        )
        return
    try:
        with psycopg.connect(settings.database.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(ENABLE_EXTENSION)
                LOGGER.info("PGvector extension enabled.")
                cur.execute(SQL_CHUNKS)
                LOGGER.info("Chunks table initialized.")
                cur.execute(SQL_DOCUMENT_CLUSTERS)
                LOGGER.info("Document clusters table initialized.")
                cur.execute(SQL_DOCUMENT_QUESTIONS)
                LOGGER.info("Document questions table initialized.")
                for migration in MIGRATIONS:
                    cur.execute(migration)
                LOGGER.info("Column migrations applied.")
                cur.execute(CREATE_INDEX_EMBEDDING)
                LOGGER.info("HNSW index on embeddings created.")
                cur.execute(CREATE_INDEX_TSVECTOR)
                LOGGER.info("GIN index on tsvector created.")
                cur.execute(CREATE_INDEX_DOC_CLUSTER_SOURCE)
                cur.execute(CREATE_INDEX_DOC_CLUSTER_TYPE)
                cur.execute(CREATE_INDEX_DOC_CLUSTER_SUPERSEDES)
                cur.execute(CREATE_INDEX_DOC_QUESTIONS_DOC)
                cur.execute(CREATE_INDEX_DOC_QUESTIONS_EMBEDDING)
                cur.execute(CREATE_INDEX_CHUNKS_DOC)
                LOGGER.info("Document cluster indexes created.")
                LOGGER.info(
                    "Cluster/medoid indexes: skipped (old K-Means path removed)."
                )

            conn.commit()
    except Exception as e:
        LOGGER.error(f"Error initializing DB: {e}")
        raise DatabaseError(f"Error initializing DB: {e}")
