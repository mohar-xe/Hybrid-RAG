"""Daily usage rate limits — 1 ingest, 5 queries per user per day.

Tracks usage in Postgres so limits survive restarts and deploys.
User identity: hashed API key, or client IP when unauthenticated.
"""

import hashlib
from datetime import date

import psycopg

from config.settings import get_settings
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)

SQL_TABLE = """
CREATE TABLE IF NOT EXISTS usage_log (
    id UUID PRIMARY KEY,
    identifier TEXT NOT NULL,
    action TEXT NOT NULL,
    day DATE NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    UNIQUE (identifier, action, day)
)"""

SQL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_usage_lookup ON usage_log (identifier, action, day)
"""

LIMITS = {"ingest": 1, "query": 5}


def init_usage_table():
    s = get_settings()
    try:
        with psycopg.connect(s.database.conninfo) as conn:
            conn.execute(SQL_TABLE)
            conn.execute(SQL_INDEX)
            conn.commit()
    except Exception as e:
        LOGGER.warning("Failed to init usage_log table: %s", e)


def _user_key(x_api_key: str | None, x_forwarded_for: str | None, remote: str) -> str:
    if x_api_key:
        return "key:" + hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
    ip = (x_forwarded_for or "").split(",")[0].strip() or remote or "unknown"
    return "ip:" + ip


def check_limit(
    action: str,
    x_api_key: str | None = None,
    x_forwarded_for: str | None = None,
    remote: str = "",
) -> tuple[bool, int]:
    """Check whether ``action`` is within today's limit for this user.

    Returns ``(allowed, used)``. The caller should raise HTTPException(429)
    when ``allowed`` is False.
    """
    limit = LIMITS.get(action)
    if limit is None:
        return True, 0

    identity = _user_key(x_api_key, x_forwarded_for, remote)
    today = date.today()
    s = get_settings()

    try:
        with psycopg.connect(s.database.conninfo) as conn:
            row = conn.execute(
                "SELECT count FROM usage_log WHERE identifier = %s AND action = %s AND day = %s",
                (identity, action, today),
            ).fetchone()

            if row and row[0] >= limit:
                LOGGER.info(
                    "Rate limit hit: %s/%s for %s (used %d)",
                    action,
                    limit,
                    identity,
                    row[0],
                )
                return False, row[0]

            if row:
                conn.execute(
                    "UPDATE usage_log SET count = count + 1 WHERE identifier = %s AND action = %s AND day = %s",
                    (identity, action, today),
                )
                used = row[0] + 1
            else:
                conn.execute(
                    "INSERT INTO usage_log (id, identifier, action, day, count) VALUES (gen_random_uuid(), %s, %s, %s, 1)",
                    (identity, action, today),
                )
                used = 1
            conn.commit()
            return True, used
    except Exception as e:
        LOGGER.warning("Rate limit check failed (allowing): %s", e)
        return True, 0
