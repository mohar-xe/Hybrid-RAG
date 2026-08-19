"""Embedding generation with a pluggable backend and API-first fallback.

``embedder(texts) -> list[vectors]`` is the single public entry point used by
both ingestion (document vectors) and the query paths (query vectors). The
backend is chosen by ``EMBEDDING__BACKEND``:

* ``api`` (default) — remote OpenAI-compatible API (``/v1/embeddings``).
* ``ollama`` — local Ollama server (``nomic-embed-text``).
* ``sentence_transformers`` — runs ``nomic-ai/nomic-embed-text-v1.5`` in-process.

On failure the configured fallback is tried (default: ollama).
"""

import hashlib
import numpy as np

from constants.logger import setup_logger
from constants.exceptions import EmbeddingError
from config.settings import EmbeddingSettings
from models.client import ApiClient, OllamaClient
from models.fallback import with_fallback
from models.rate_limiter import RateLimiter
from cache import embedding_cache

LOGGER = setup_logger(__name__)
EMBEDDING_DIM = 256
# nomic-embed-text has a 2048-token context. ~4 chars/token, so ~8000 chars is
# a safe per-input budget. We cap client-side (deterministic, fast, and avoids
# huge request payloads) rather than relying solely on server-side truncation.
MAX_INPUT_CHARS = 8000

_settings = EmbeddingSettings()

# Paces API embedding requests (``EMBEDDING__API_RATE_LIMIT``, default 1 RPM).
# Free-tier keys throttle hard (Mistral 429s above a small burst); batching keeps
# request counts low, and this limiter keeps the burst under the provider's cap.
_API_LIMITER = RateLimiter(calls_per_minute=_settings.api_rate_limit)

# Lazily-instantiated in-process sentence-transformers model (deployment path).
# Module-level cache so the (heavy) model is loaded at most once per process.
_ST_MODEL = None


def ensure_model(model: str) -> None:
    """Pull the Ollama embedding model if it isn't present (no-op if it is)."""
    import ollama

    available = {m.model for m in ollama.list().models}
    if model not in available and f"{model}:latest" not in available:
        LOGGER.info(f"Pulling model '{model}'...")
        ollama.pull(model)


def _cap(texts: list[str], prefix: str) -> list[str]:
    """Apply the optional task prefix and cap each input to the context budget."""
    if prefix:
        return [(prefix + t)[:MAX_INPUT_CHARS] for t in texts]
    return [t[:MAX_INPUT_CHARS] for t in texts]


def _log_truncated(chunks: list[str]) -> None:
    truncated = sum(1 for c in chunks if len(c) > MAX_INPUT_CHARS)
    if truncated:
        LOGGER.warning(
            f"{truncated}/{len(chunks)} input(s) exceeded {MAX_INPUT_CHARS} chars "
            f"and were truncated before embedding."
        )


def _embed_api(chunks: list[str], model: str, dim: int, batch_size: int) -> np.ndarray:
    api_key = _settings.api_key.get_secret_value()
    api_model = _settings.api_model or model
    if not api_key or not _settings.api_base_url:
        raise EmbeddingError(
            "API embedding backend requires EMBEDDING__API_BASE_URL and EMBEDDING__API_KEY."
        )
    client = ApiClient(base_url=_settings.api_base_url, api_key=api_key)
    _log_truncated(chunks)
    inputs = _cap(chunks, _settings.query_prefix)
    all_embeddings: list[list[float]] = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        _API_LIMITER.acquire()
        emb = client.embed(batch, model=api_model)
        all_embeddings.extend(emb)
    return np.array(all_embeddings)[:, :dim]


def _embed_ollama(
    chunks: list[str], model: str, dim: int, batch_size: int
) -> np.ndarray:
    import ollama as _ollama

    ensure_model(model)
    _log_truncated(chunks)

    all_embeddings: list[list[float]] = []
    inputs = _cap(chunks, _settings.query_prefix)
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        response = _ollama.embed(model, input=batch, truncate=True)
        all_embeddings.extend(response["embeddings"])

    return np.array(all_embeddings)[:, :dim]


def _get_st_model():
    """Load (once) and return the in-process sentence-transformers model."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer

        LOGGER.info(
            f"Loading sentence-transformers embedder '{_settings.st_model}' (CPU)..."
        )
        _ST_MODEL = SentenceTransformer(
            _settings.st_model, trust_remote_code=True, device="cpu"
        )
    return _ST_MODEL


def _embed_sentence_transformers(
    chunks: list[str], dim: int, batch_size: int
) -> np.ndarray:
    model = _get_st_model()
    _log_truncated(chunks)

    inputs = _cap(chunks, _settings.query_prefix)
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(vecs)[:, :dim]


_BACKENDS = {
    "api": _embed_api,
    "ollama": _embed_ollama,
    "sentence_transformers": _embed_sentence_transformers,
}


def _run_backend(name: str, chunks: list[str], dim: int, batch_size: int) -> np.ndarray:
    fn = _BACKENDS.get(name)
    if fn is None:
        raise EmbeddingError(f"Unknown embedding backend: {name!r}")
    # sentence_transformers backend manages its own model internally; passing
    # ``model=`` to it would cause a TypeError.
    if name == "sentence_transformers":
        return fn(chunks, dim=dim, batch_size=batch_size)
    return fn(chunks, model=_settings.model, dim=dim, batch_size=batch_size)


def embedder(
    chunks: list[str],
    model: str | None = None,
    dim: int = EMBEDDING_DIM,
    batch_size: int | None = None,
) -> list[list[float]]:
    """Embed ``chunks`` into ``dim``-d vectors using the configured backend.

    Tries the primary backend first; on failure falls back to the configured
    fallback (if enabled). Returns the vector list only (backend tag is logged).
    """
    if not chunks:
        return []

    _cache_key: str | None = None
    if len(chunks) == 1:
        _cache_key = hashlib.md5(chunks[0].encode("utf-8")).hexdigest()
        cached = embedding_cache.get(_cache_key)
        if cached is not None:
            LOGGER.debug("Embedding cache hit for %s...", chunks[0][:40])
            return cached

    batch_size = batch_size or _settings.batch_size

    primary = _settings.backend

    if _settings.fallback_enabled and primary != _settings.fallback_backend:

        def primary_fn(texts=chunks):
            return _run_backend(primary, texts, dim, batch_size)

        def fallback_fn(texts=chunks):
            return _run_backend(_settings.fallback_backend, texts, dim, batch_size)

        try:
            embeddings, backend = with_fallback(
                primary_fn,
                fallback_fn,
                "embedder",
                primary_tag=primary,
                fallback_tag=_settings.fallback_backend,
            )
        except Exception as e:
            raise EmbeddingError(f"Embedding failed (all backends): {e}")
    else:
        try:
            embeddings = _run_backend(primary, chunks, dim, batch_size)
            backend = primary
        except Exception as e:
            raise EmbeddingError(f"Embedding failed (backend={primary}): {e}")

    result = embeddings.tolist()
    if _cache_key is not None:
        embedding_cache.set(_cache_key, result)
    LOGGER.info(f"Embedded {len(chunks)} chunks (dim={dim}, backend={backend}).")
    return result
