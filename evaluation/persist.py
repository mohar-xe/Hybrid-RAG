"""JSON persistence for the staged evaluation pipeline.

The eval harness runs in decoupled, resumable phases, each persisted under
``config.CACHE_DIR``:

    artifacts  -> query_embeddings_n{n}_s{seed}.json   (1 Mistral batched call)
                  query_entities_n{n}_s{seed}.json     (1 Gemini bundled call)
    retrieve   -> retrievals_n{n}_s{seed}.json         (final reranked chunks)
    generate   -> answers_n{n}_s{seed}.json            (bundled generation output)

Every cache entry is keyed by question id (and, for retrieval/answers, by the
run-config label like ``all_three+rerank``). ``load`` returns ``None`` when the
phase file does not exist; ``--force`` recomputes a phase by clearing its file
first. The whole cache can be wiped with ``clear``.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation import config


def _stem(kind: str, n: int, seed: int) -> str:
    return f"{kind}_n{n}_s{seed}"


def cache_path(kind: str, n: int = config.N_QUERIES, seed: int = config.SEED) -> Path:
    """Path of the JSON file for a phase (kind in the module docstring)."""
    return config.CACHE_DIR / f"{_stem(kind, n, seed)}.json"


def save(data: dict, kind: str, n: int = config.N_QUERIES, seed: int = config.SEED) -> Path:
    """Atomically persist a phase dict to its cache file."""
    path = cache_path(kind, n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load(kind: str, n: int = config.N_QUERIES, seed: int = config.SEED) -> dict | None:
    """Load a phase dict, or ``None`` if the phase has not run."""
    path = cache_path(kind, n, seed)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def clear(kind: str | None = None, n: int = config.N_QUERIES, seed: int = config.SEED) -> int:
    """Delete one phase file (or the whole cache dir when ``kind`` is None).

    Returns the number of files removed.
    """
    removed = 0
    if kind is None:
        if config.CACHE_DIR.exists():
            for p in config.CACHE_DIR.iterdir():
                p.unlink()
                removed += 1
        return removed
    path = cache_path(kind, n, seed)
    if path.exists():
        path.unlink()
        removed = 1
    return removed