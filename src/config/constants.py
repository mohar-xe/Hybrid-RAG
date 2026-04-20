import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NER_MODEL = "google/gemma-4-27b-it"
GENERATOR_MODEL = "google/gemma-4-27b-it"


def _build_db_conn() -> str:
    direct = os.getenv("DB_CONN", "").strip()
    if direct:
        return direct

    db = os.getenv("DB", "").strip()
    user = os.getenv("DB_USER", "").strip()
    password = os.getenv("DB_PASSWORD", "").strip()
    host = os.getenv("DB_HOST", "localhost").strip() or "localhost"
    port = os.getenv("DB_PORT", "5432").strip() or "5432"

    if not (db and user and password):
        return ""

    return f"dbname={db} user={user} password={password} host={host} port={port}"


DB_CONN = _build_db_conn()


def get_db_conn() -> str:
    if not DB_CONN:
        raise RuntimeError(
            "Database connection is not configured. Set DB_CONN or DB, DB_USER, DB_PASSWORD."
        )
    return DB_CONN