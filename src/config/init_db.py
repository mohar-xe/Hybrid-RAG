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

def init_db():
    try:
        with psycopg.connect(f"host={settings.database.host} port={settings.database.port} dbname={settings.database.db_name} user={settings.database.user} password={settings.database.password.get_secret_value()}") as conn:
            with conn.cursor() as cur: 
                cur.execute(ENABLE_EXTENSION)
                LOGGER.info("PGvector extension enabled.")
                cur.execute(SQL_SCHEMA)
                LOGGER.info("Schema skeleton initialized.")
                cur.execute(CREATE_INDEX_EMBEDDING)
                LOGGER.info("HNSW Index at embeddings.")
                cur.execute(CREATE_INDEX_TSVECTOR)
                LOGGER.info("GIN Index at keywordss.")

            conn.commit()
    except Exception as e:
        LOGGER.error(f"Error initializing DB: {e}")
        raise DatabaseError(f"Error initializing DB: {e}")