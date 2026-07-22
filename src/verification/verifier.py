"""Faithfulness verification via NLI / LLM-as-verifier (gated by VERIFIER__ENABLED).

Scores how well a generated answer is *entailed* by the retrieved context. Three
backends:

* ``api``    — remote NLI/entailment API endpoint (default).
* ``ollama`` — LLM-as-verifier via a prompting template.
* ``hf``    — local HuggingFace NLI cross-encoder.

On failure the configured fallback is tried (default: ollama). Only runs when
``VERIFIER__ENABLED`` is true.
"""

from functools import lru_cache

import numpy as np

from config.settings import get_settings
from constants.logger import setup_logger
from models.client import ApiClient, OllamaClient, HFClient
from models.fallback import with_fallback

LOGGER = setup_logger(__name__)
settings = get_settings()

ENTAILMENT_IDX = (
    1  # cross-encoder/nli-deberta-v3-base: 0=contradiction, 1=entailment, 2=neutral
)

_VERIFY_PROMPT = (
    "Does the evidence support the claim? "
    "Answer with exactly one word: entailment, contradiction, or neutral."
)


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence split (no NLTK dependency): break on ., !, ? and newlines."""
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# --- HF backend ---


@lru_cache(maxsize=1)
def _hf_model():
    from sentence_transformers import CrossEncoder

    name = settings.verifier.model
    LOGGER.info(f"Loading NLI verifier '{name}'...")
    return CrossEncoder(name)


def _verify_hf(answer: str, context: str) -> float | None:
    if not answer.strip() or not context.strip():
        return None
    sentences = _split_sentences(answer)
    if not sentences:
        return None
    model = _hf_model()
    pairs = [(context, sentence) for sentence in sentences]
    logits = model.predict(pairs)
    probs = _softmax(logits)
    entailment = probs[:, ENTAILMENT_IDX]
    return float(np.mean(entailment))


# --- API backend ---


def _verify_api(answer: str, context: str) -> float | None:
    if not answer.strip() or not context.strip():
        return None
    api_key = settings.verifier.api_key.get_secret_value()
    if not api_key or not settings.verifier.api_base_url:
        raise RuntimeError("Verifier API not configured")
    client = ApiClient(
        base_url=settings.verifier.api_base_url,
        api_key=api_key,
        timeout=30.0,
    )
    sentences = _split_sentences(answer)
    if not sentences:
        return None
    scores: list[float] = []
    for sentence in sentences:
        prompt = f"Evidence: {context}\n\nClaim: {sentence}\n\n{_VERIFY_PROMPT}"
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.verifier.api_model or "deepseek-v4-flash",
            temperature=0.0,
            max_tokens=10,
        )
        response = response.strip().lower()
        if "entailment" in response:
            scores.append(1.0)
        elif "contradiction" in response:
            scores.append(0.0)
        else:
            scores.append(0.5)  # neutral
    return float(np.mean(scores)) if scores else None


# --- Ollama backend ---


def _verify_ollama(answer: str, context: str) -> float | None:
    if not answer.strip() or not context.strip():
        return None
    client = OllamaClient(base_url=settings.verifier.ollama_base_url)
    sentences = _split_sentences(answer)
    if not sentences:
        return None
    scores: list[float] = []
    for sentence in sentences:
        prompt = f"Evidence: {context}\n\nClaim: {sentence}\n\n{_VERIFY_PROMPT}"
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.verifier.ollama_model,
            options={"temperature": 0.0},
        )
        response = response.strip().lower()
        if "entailment" in response:
            scores.append(1.0)
        elif "contradiction" in response:
            scores.append(0.0)
        else:
            scores.append(0.5)
    return float(np.mean(scores)) if scores else None


# --- Public entry point ---


def score_faithfulness(answer: str, context: str) -> float | None:
    """Return mean entailment probability of ``answer`` given ``context``.

    Tries the primary backend first; on failure falls back to the configured
    fallback. Returns ``None`` when there is nothing meaningful to score, or
    when all backends fail (the caller is expected to guard against this).
    """
    if not settings.verifier.enabled:
        return None

    backend = settings.verifier.backend
    fallback = settings.verifier.fallback_backend
    fallback_enabled = settings.verifier.fallback_enabled

    _BACKENDS = {
        "api": _verify_api,
        "ollama": _verify_ollama,
        "hf": _verify_hf,
    }

    primary_fn = _BACKENDS.get(backend, _verify_hf)
    if fallback_enabled and fallback != backend:
        fallback_fn = _BACKENDS.get(fallback, _verify_hf)
        try:
            result, _tag = with_fallback(
                primary_fn,
                fallback_fn,
                "verifier",
                fallback_enabled=True,
                primary_tag=backend,
                fallback_tag=fallback,
                answer=answer,
                context=context,
            )
        except Exception as exc:
            LOGGER.warning("All verifier backends failed: %s", exc)
            return None
    else:
        try:
            result = primary_fn(answer=answer, context=context)
        except Exception as exc:
            LOGGER.warning("Verifier backend %s failed: %s", backend, exc)
            return None

    if result is not None:
        LOGGER.info(
            "Faithfulness %.3f (threshold %s).",
            result,
            settings.verifier.threshold,
        )
    return result
