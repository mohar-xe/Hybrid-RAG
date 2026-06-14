import ollama
import numpy as np

from constants.logger import setup_logger
from constants.exceptions import EmbeddingError

LOGGER = setup_logger(__name__)
EMBEDDING_DIM = 256
BATCH_SIZE = 128


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

    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = ollama.embed(model, input=batch)
        all_embeddings.extend(response["embeddings"])

    embeddings = np.array(all_embeddings)[:, :dim]
    LOGGER.info(f"Embedded {len(chunks)} chunks (dim={dim}).")
    return embeddings.tolist()
