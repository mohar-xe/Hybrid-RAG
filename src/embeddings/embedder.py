"""Embedding generation with a pluggable backend.

``embedder(texts) -> list[vectors]`` is the single public entry point used by
both ingestion (document vectors) and the query paths (query vectors). The
backend is chosen by ``EMBEDDING__BACKEND``:

* ``ollama`` (default) — calls a local Ollama server (``nomic-embed-text``).
  Used for ingestion and local development.
* ``sentence_transformers`` — runs ``nomic-ai/nomic-embed-text-v1.5`` in-process
  (CPU), no Ollama required. Intended for the read-only / free deployment.

Both paths emit the **same** representation as the frozen corpus: raw text (no
task prefix, unless ``EMBEDDING__QUERY_PREFIX`` is set), capped to
``MAX_INPUT_CHARS``, and truncated to the first ``dim`` (256) dimensions
(Matryoshka). Vectors are returned un-normalized — pgvector cosine (``<=>``) is
scale-invariant, so query normalization does not affect ranking.
"""

import numpy as np

from constants.logger import setup_logger
from constants.exceptions import EmbeddingError
from config.settings import EmbeddingSettings

LOGGER = setup_logger(__name__)
EMBEDDING_DIM = 256
# Ollama CPU embedding shows no batch speedup (per-input cost is fixed) and a
# very large batch becomes a single long-running request that monopolizes the
# runner. A modest batch keeps requests short and progress incremental.
BATCH_SIZE = 16
# nomic-embed-text has a 2048-token context. ~4 chars/token, so ~8000 chars is
# a safe per-input budget. We cap client-side (deterministic, fast, and avoids
# huge request payloads) rather than relying solely on server-side truncation.
MAX_INPUT_CHARS = 8000

_settings = EmbeddingSettings()

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


def _embed_ollama(chunks: list[str], model: str, dim: int, batch_size: int) -> np.ndarray:
    import ollama

    ensure_model(model)

    truncated = sum(1 for c in chunks if len(c) > MAX_INPUT_CHARS)
    if truncated:
        LOGGER.warning(
            f"{truncated}/{len(chunks)} input(s) exceeded {MAX_INPUT_CHARS} chars "
            f"and were truncated before embedding."
        )

    all_embeddings: list[list[float]] = []
    inputs = _cap(chunks, _settings.query_prefix)
    for i in range(0, len(inputs), batch_size):
        # `truncate=True` is a server-side safety net for token-vs-char slack.
        batch = inputs[i : i + batch_size]
        response = ollama.embed(model, input=batch, truncate=True)
        all_embeddings.extend(response["embeddings"])

    return np.array(all_embeddings)[:, :dim]


def _get_st_model():
    """Load (once) and return the in-process sentence-transformers model."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer

        LOGGER.info(f"Loading sentence-transformers embedder '{_settings.st_model}' (CPU)...")
        # nomic-embed-text-v1.5 ships custom modeling code -> trust_remote_code.
        _ST_MODEL = SentenceTransformer(_settings.st_model, trust_remote_code=True, device="cpu")
    return _ST_MODEL


def _embed_sentence_transformers(chunks: list[str], dim: int, batch_size: int) -> np.ndarray:
    model = _get_st_model()

    truncated = sum(1 for c in chunks if len(c) > MAX_INPUT_CHARS)
    if truncated:
        LOGGER.warning(
            f"{truncated}/{len(chunks)} input(s) exceeded {MAX_INPUT_CHARS} chars "
            f"and were truncated before embedding."
        )

    inputs = _cap(chunks, _settings.query_prefix)
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    # Matryoshka truncation to match the stored 256-d vectors.
    return np.asarray(vecs)[:, :dim]


def embedder(
    chunks: list[str],
    model: str | None = None,
    dim: int = EMBEDDING_DIM,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """Embed ``chunks`` into ``dim``-d vectors using the configured backend."""
    if not chunks:
        return []

    try:
        if _settings.backend == "sentence_transformers":
            embeddings = _embed_sentence_transformers(chunks, dim, batch_size)
        else:
            embeddings = _embed_ollama(chunks, model or _settings.model, dim, batch_size)
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Embedding failed (backend={_settings.backend}): {e}")

    LOGGER.info(f"Embedded {len(chunks)} chunks (dim={dim}, backend={_settings.backend}).")
    return embeddings.tolist()
