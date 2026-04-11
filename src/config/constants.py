import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embed-v1"
NER_MODEL = "meta/llama-3.1-8b-instruct"
GENERATOR_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


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