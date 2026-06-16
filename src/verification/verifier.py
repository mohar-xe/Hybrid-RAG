"""Faithfulness verification via an NLI cross-encoder (gated).

Scores how well a generated answer is *entailed* by the retrieved context. This
is the implementation behind the ``faithfulness`` field on ``/query`` responses;
it only runs when ``VERIFIER__ENABLED`` is true, because the NLI model is heavy
and adds latency to every request. The model is loaded lazily and cached,
mirroring ``retrieval.reranker``.

Score semantics: each answer sentence is treated as a hypothesis and the full
retrieved context as the premise. We take the model's entailment probability per
sentence and average them, yielding a value in ``[0, 1]`` (higher = better
grounded). ``settings.verifier.threshold`` is the suggested accept cutoff.
"""

from functools import lru_cache

import numpy as np

from config.settings import get_settings
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()

# Label order emitted by cross-encoder/nli-deberta-v3-base:
#   index 0 = contradiction, 1 = entailment, 2 = neutral
_ENTAILMENT_IDX = 1


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    name = settings.verifier.model
    LOGGER.info(f"Loading NLI verifier '{name}'...")
    return CrossEncoder(name)


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


def score_faithfulness(answer: str, context: str) -> float | None:
    """Return mean entailment probability of ``answer`` given ``context``.

    Returns ``None`` when there is nothing meaningful to score (empty answer or
    context). The caller is expected to guard model-loading failures.
    """
    if not answer.strip() or not context.strip():
        return None

    sentences = _split_sentences(answer)
    if not sentences:
        return None

    model = _model()
    pairs = [(context, sentence) for sentence in sentences]
    logits = model.predict(pairs)
    probs = _softmax(logits)
    entailment = probs[:, _ENTAILMENT_IDX]

    score = float(np.mean(entailment))
    LOGGER.info(
        f"Faithfulness {score:.3f} over {len(sentences)} answer sentence(s) "
        f"(threshold {settings.verifier.threshold})."
    )
    return score
