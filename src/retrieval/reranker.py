"""Cross-encoder reranking (Stage 4).

Re-scores retrieval candidates with a cross-encoder (query, chunk) model and
keeps the top-k. The model name and default ``top_k`` come from
``RerankerSettings`` (``RERANKER__*``). The model is loaded lazily and cached so
``--help`` and import stay fast and the (large) model only loads when reranking
actually runs.
"""

from functools import lru_cache

from config.settings import get_settings
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    name = settings.reranker.model
    LOGGER.info(f"Loading cross-encoder reranker '{name}'...")
    return CrossEncoder(name)


def rerank(query: str, chunks: list, top_k: int | None = None) -> list:
    """Re-score ``chunks`` against ``query`` and return the top-k.

    ``chunks`` are retrieval results exposing ``.text`` and a writable
    ``.score`` (e.g. ``retrieval.pgvector.RetrievedChunk``). The cross-encoder
    relevance score replaces ``.score`` on each returned chunk.
    """
    if not chunks:
        return []
    top_k = top_k or settings.reranker.top_k

    model = _model()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        chunk.score = float(score)
        out.append(chunk)

    LOGGER.info(f"Reranked {len(chunks)} candidate(s) -> top {len(out)}.")
    return out
