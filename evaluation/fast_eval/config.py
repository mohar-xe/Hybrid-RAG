"""fast_eval configuration — every knob in one easy place.

Change the values in this file, or set the matching ``FAST_EVAL_*`` environment
variables before running, to switch model / provider / API key / worker count.
Everything here is separate from the staged ``evaluation/config.py`` — that suite
stays for personal free-tier use and is untouched by this one.

The model/provider/config values below read (in order): an explicit env var →
this file's default → the project ``.env`` (for API keys only).
"""

from __future__ import annotations

import os
from pathlib import Path

_FAST_EVAL_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _FAST_EVAL_ROOT.parent.parent  # repo root

DATA_DIR = _FAST_EVAL_ROOT / "data"
RESULTS_DIR = _FAST_EVAL_ROOT / "results"   # its own dir, not the staged one's

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_QUERIES = 100
TOP_K = 5
CANDIDATE_K = 20
MAX_WORKERS = 2  # concurrency: the one tunable users change most; 2 is a safe default

# --- Generator / model / provider -------------------------------------------------
# Paid key → no free-tier throttling. fast_eval drives generation directly and can
# run every query/cell concurrently. Change these to switch provider.
GENERATOR_MODEL = "deepseek-v4-flash"
GENERATOR_BASE_URL = "https://api.b.ai/v1"
GENERATOR_API_KEY = ""          # empty → falls back to $GENERATOR__API_KEY from .env
GENERATOR_FALLBACK_ENABLED = False
# Cap per-query answer length. DeepSeek v4-flash reasons by default; capping
# tokens (and disabling thinking below) keeps extractive answers fast. 0 = send
# no max_tokens (uncapped, verbose reasoning).
GENERATOR_MAX_TOKENS = 128

# Whether to (re)ingest if the corpus is missing at check time.
INGEST_IF_MISSING = True

# Source identifier used to detect an already-ingested HotpotQA corpus.
CORPUS_SOURCE_TYPE = "hotpotqa"


def resolve_api_key() -> str:
    """Resolve the generator API key: explicit value → env var → project .env."""
    if GENERATOR_API_KEY:
        return GENERATOR_API_KEY
    env_key = os.environ.get("GENERATOR__API_KEY", "")
    if env_key:
        return env_key
    env_file = _PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GENERATOR__API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""
