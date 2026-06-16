import ollama
import numpy as np

from constants.logger import setup_logger
from constants.exceptions import EmbeddingError

LOGGER = setup_logger(__name__)
EMBEDDING_DIM = 256
# Ollama CPU embedding shows no batch speedup (per-input cost is fixed) and a
# very large batch becomes a single long-running request that monopolizes the
# runner. A modest batch keeps requests short and progress incremental.
BATCH_SIZE = 16
# nomic-embed-text has a 2048-token context. ~4 chars/token, so ~8000 chars is
# a safe per-input budget. We cap client-side (deterministic, fast, and avoids
# huge request payloads) rather than relying solely on Ollama's server-side
# truncation, which still 400s when a *batch* of over-long inputs is sent.
MAX_INPUT_CHARS = 8000


def ensure_model(model: str) -> None:
    available = {m.model for m in ollama.list().models}
    if model not in available and f"{model}:latest" not in available:
        LOGGER.info(f"Pulling model '{model}'...")
        ollama.pull(model)


def embedder(
    chunks: list[str],
    model: str = "nomic-embed-text",
    dim: int = EMBEDDING_DIM,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    if not chunks:
        return []

    ensure_model(model)

    truncated = sum(1 for c in chunks if len(c) > MAX_INPUT_CHARS)
    if truncated:
        LOGGER.warning(
            f"{truncated}/{len(chunks)} input(s) exceeded {MAX_INPUT_CHARS} chars "
            f"and were truncated before embedding."
        )

    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        # Cap each input to the model context budget. `truncate=True` is kept as
        # a server-side safety net for the remaining token-vs-char slack.
        batch = [c[:MAX_INPUT_CHARS] for c in chunks[i : i + batch_size]]
        response = ollama.embed(model, input=batch, truncate=True)
        all_embeddings.extend(response["embeddings"])

    embeddings = np.array(all_embeddings)[:, :dim]
    LOGGER.info(f"Embedded {len(chunks)} chunks (dim={dim}).")
    return embeddings.tolist()
