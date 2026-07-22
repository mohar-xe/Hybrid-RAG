"""Simple in-memory LRU caches for expensive operations.

Usage::

    from cache import embedding_cache, metadata_cache, query_cache

    # embedding_cache caches list[list[float]] keyed by tuple[str] (batches)
    # metadata_cache caches dict keyed by str (text hash)
    # query_cache caches dict keyed by str (question text)
"""

from collections import OrderedDict
from typing import Any


class Cache:
    """LRU cache with max size. Not thread-safe — callers own the concurrency."""

    def __init__(self, maxsize: int = 128):
        self._maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Shared cache instances used across the pipeline
# ---------------------------------------------------------------------------

# Embedding cache: key = hash of the joined input texts, value = vectors.
# The embedder is called per-query and per-metadata-step; caching exact repeats
# (e.g. the same query embedding requested twice) avoids the ~200 ms Ollama call.
embedding_cache = Cache(maxsize=256)

# Document metadata cache: key = md5 of the first 500 chars of document text.
# Metadata extraction is a Gemini API call or spaCy pass; caching avoids
# re-extracting when re-ingesting the same doc (e.g. --no-store --graph retry).
metadata_cache = Cache(maxsize=64)

# Query interpretation cache: key = normalized question text.
# interpret_query hits the Flash-Lite API; caching avoids repeated API calls
# for the same question during development / debugging.
query_cache = Cache(maxsize=128)
