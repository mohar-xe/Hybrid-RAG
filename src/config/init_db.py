import psycopg

from constants.logger import setup_logger
from constants.exceptions import DatabaseError
from config.settings import get_settings

LOGGER = setup_logger(__name__)
settings = get_settings()

ENABLE_EXTENSION = """CREATE EXTENSION IF NOT EXISTS vector"""

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
chunk_id TEXT PRIMARY KEY,
text TEXT NOT NULL,
embedding vector(256),
source_type TEXT NOT NULL,
source_id TEXT NOT NULL,
chunk_index INTEGER NOT NULL,
keyword TEXT[],
tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
)"""

CREATE_INDEX_EMBEDDING = """CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"""
CREATE_INDEX_TSVECTOR = """CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv)"""

# Idempotent migrations for columns added after the initial schema. Stage 2
# (clustering) needs cluster_id + is_medoid; coarse->fine retrieval filters and
# routes on them. ADD COLUMN IF NOT EXISTS makes this safe on existing tables.
MIGRATIONS = [
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS cluster_id INTEGER",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_medoid BOOLEAN NOT NULL DEFAULT FALSE",
]
# Partial index over medoids only (the coarse-pass search space) + a cluster_id
# index for the fine, cluster-filtered pass.
CREATE_INDEX_MEDOID = """CREATE INDEX IF NOT EXISTS idx_chunks_medoid ON chunks (is_medoid) WHERE is_medoid"""
CREATE_INDEX_CLUSTER = """CREATE INDEX IF NOT EXISTS idx_chunks_cluster ON chunks (cluster_id)"""

def init_db():
    if not settings.database.init_schema:
        LOGGER.info("DATABASE__INIT_SCHEMA is false — skipping schema init (read-only mode).")
        return
    try:
        with psycopg.connect(settings.database.conninfo) as conn:
            with conn.cursor() as cur: 
                cur.execute(ENABLE_EXTENSION)
                LOGGER.info("PGvector extension enabled.")
                cur.execute(SQL_SCHEMA)
                LOGGER.info("Schema skeleton initialized.")
                for migration in MIGRATIONS:
                    cur.execute(migration)
                LOGGER.info("Column migrations applied.")
                cur.execute(CREATE_INDEX_EMBEDDING)
                LOGGER.info("HNSW index on embeddings created.")
                cur.execute(CREATE_INDEX_TSVECTOR)
                LOGGER.info("GIN index on tsvector created.")
                cur.execute(CREATE_INDEX_MEDOID)
                cur.execute(CREATE_INDEX_CLUSTER)
                LOGGER.info("Cluster/medoid indexes created.")

            conn.commit()
    except Exception as e:
        LOGGER.error(f"Error initializing DB: {e}")
        raise DatabaseError(f"Error initializing DB: {e}")