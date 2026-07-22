"""Cross-encoder reranking (Stage 4) with API-first + fallback.

Re-scores retrieval candidates with a cross-encoder (query, chunk) model and
keeps the top-k. Three backends:
* ``api``    — remote cross-encoder API endpoint (default).
* ``ollama`` — LLM-as-reranker via a scoring prompt.
* ``hf``    — local HuggingFace cross-encoder.

On failure the configured fallback is tried (default: ollama).
"""

from functools import lru_cache

from config.settings import get_settings
from constants.logger import setup_logger
from models.client import ApiClient, OllamaClient, HFClient
from models.fallback import with_fallback

LOGGER = setup_logger(__name__)
settings = get_settings()

_RERANK_PROMPT = (
    "On a scale of 0-10, how relevant is this text to the query? "
    "Only output a single number and nothing else."
)


@lru_cache(maxsize=1)
def _hf_model():
    from sentence_transformers import CrossEncoder

    name = settings.reranker.model
    LOGGER.info(f"Loading cross-encoder reranker '{name}'...")
    return CrossEncoder(name)


def _rerank_hf(query: str, chunks: list, top_k: int) -> list:
    """HF cross-encoder reranking (the original local implementation)."""
    if not chunks:
        return []
    model = _hf_model()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        chunk.score = float(score)
        out.append(chunk)
    return out


def _rerank_api(query: str, chunks: list, top_k: int) -> list:
    """Remote API reranking via a single prompt with per-chunk scoring.

    Sends all chunks in one request with a scored-line format (``0: 8``,
    ``1: 3``, …) that is easy to parse regardless of the model's verbosity.
    """
    if not chunks:
        return []
    api_key = settings.reranker.api_key.get_secret_value()
    if not api_key or not settings.reranker.api_base_url:
        LOGGER.warning("Reranker API not configured; falling back.")
        raise RuntimeError("Reranker API not configured")
    client = ApiClient(
        base_url=settings.reranker.api_base_url,
        api_key=api_key,
        timeout=30.0,
    )

    lines = []
    for i, c in enumerate(chunks):
        text = c.text[:500].replace("\n", " ")
        lines.append(f"[{i}] {text}")

    prompt = (
        f"Query: {query}\n\n"
        f"Rate each passage for relevance to the query (0 = irrelevant, "
        f"10 = highly relevant).\n\n"
        f"Passages:\n" + "\n".join(lines) + "\n\n"
        f"Return one score per line: the index, a colon, and the number.\n"
        f"Example:\n0: 7\n1: 3\n2: 9"
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.reranker.api_model or "deepseek-v4-flash",
        temperature=0.0,
        max_tokens=len(chunks) * 10,
    )

    import re

    scores: dict[int, float] = {}
    for line in response.strip().split("\n"):
        m = re.match(r"(\d+)\s*:\s*([\d.]+)", line.strip())
        if m:
            idx = int(m.group(1))
            score = float(m.group(2))
            if 0 <= idx < len(chunks):
                scores[idx] = score

    if not scores:
        raise RuntimeError(f"Could not parse reranker scores from: {response[:200]}")

    all_scores = [scores.get(i, 0.0) for i in range(len(chunks))]

    ranked = sorted(zip(chunks, all_scores), key=lambda cs: cs[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        chunk.score = float(score)
        out.append(chunk)
    return out


def _rerank_ollama(query: str, chunks: list, top_k: int) -> list:
    """LLM-as-reranker via Ollama: prompt the model to score each chunk."""
    if not chunks:
        return []
    client = OllamaClient(base_url=settings.reranker.ollama_base_url)
    scores: list[float] = []
    for chunk in chunks:
        prompt = f"Query: {query}\n\nText: {chunk.text}\n\n{_RERANK_PROMPT}"
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.reranker.ollama_model,
            options={"temperature": 0.0},
        )
        try:
            score = float(response.strip())
        except (ValueError, TypeError):
            score = 0.0
        scores.append(score)

    ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        chunk.score = float(score)
        out.append(chunk)
    return out


def _rerank_jina(query: str, chunks: list, top_k: int) -> list:
    """Jina AI dedicated reranker API."""
    if not chunks:
        return []
    import httpx

    api_key = settings.reranker.api_key.get_secret_value()
    model = settings.reranker.api_model or "jina-reranker-v2-base-en"
    texts = [c.text for c in chunks]

    resp = httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "query": query, "documents": texts, "top_n": top_k},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    scored = {}
    for r in results:
        idx = r["index"]
        scored[idx] = r["relevance_score"]

    out = []
    for r in results:
        idx = r["index"]
        chunk = chunks[idx]
        chunk.score = float(scored.get(idx, 0.0))
        out.append(chunk)
    return out


def rerank(query: str, chunks: list, top_k: int | None = None) -> list:
    """Re-score ``chunks`` against ``query`` and return the top-k.

    Tries the primary backend first; on failure falls back to the configured
    fallback (if enabled). Returns the reranked chunk list only (backend tag is
    logged internally).
    """
    if not chunks:
        return []
    top_k = top_k or settings.reranker.top_k

    backend = settings.reranker.backend
    fallback = settings.reranker.fallback_backend
    fallback_enabled = settings.reranker.fallback_enabled

    _BACKENDS = {
        "api": _rerank_api,
        "jina": _rerank_jina,
        "ollama": _rerank_ollama,
        "hf": _rerank_hf,
    }

    primary_fn = _BACKENDS.get(backend)
    if primary_fn is None:
        LOGGER.warning("Unknown reranker backend %r, falling back to hf.", backend)
        primary_fn = _rerank_hf
        backend = "hf"

    if fallback_enabled and fallback != backend:
        fallback_fn = _BACKENDS.get(fallback, _rerank_hf)
        try:
            result, _tag = with_fallback(
                primary_fn,
                fallback_fn,
                "reranker",
                fallback_enabled=True,
                primary_tag=backend,
                fallback_tag=fallback,
                query=query,
                chunks=chunks,
                top_k=top_k,
            )
        except Exception:
            LOGGER.warning(
                "All reranker backends failed; returning un-reranked chunks."
            )
            return chunks[:top_k]
    else:
        try:
            result = primary_fn(query=query, chunks=chunks, top_k=top_k)
        except Exception as e:
            LOGGER.warning(
                "Reranker backend %s failed: %s; returning un-reranked.", backend, e
            )
            return chunks[:top_k]

    LOGGER.info(f"Reranked {len(chunks)} candidate(s) -> top {len(result)}.")
    return result
