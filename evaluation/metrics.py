"""Answer, retrieval, and latency metrics.

All functions here are **pure** and depend only on the standard library, so they
can be unit-tested without a database, models, or network. Answer metrics follow
the standard SQuAD / HotpotQA normalization (lowercase, strip punctuation and
articles, collapse whitespace).
"""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation/articles, squeeze ws."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, gold: str) -> float:
    """1.0 iff the normalized prediction equals the normalized gold answer."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def _overlap(prediction: str, gold: str) -> tuple[int, int, int]:
    """Return (num_overlapping_tokens, n_pred_tokens, n_gold_tokens)."""
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(gold)
    common = Counter(pred_toks) & Counter(gold_toks)
    return sum(common.values()), len(pred_toks), len(gold_toks)


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold (HotpotQA's primary metric)."""
    num_same, n_pred, n_gold = _overlap(prediction, gold)
    if n_pred == 0 and n_gold == 0:
        return 1.0  # both empty (e.g. both normalize away) -> trivially equal
    if num_same == 0 or n_pred == 0 or n_gold == 0:
        return 0.0
    precision = num_same / n_pred
    recall = num_same / n_gold
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, gold: str) -> float:
    """Fraction of gold answer tokens recovered by the prediction."""
    num_same, n_pred, n_gold = _overlap(prediction, gold)
    if n_gold == 0:
        return 1.0 if n_pred == 0 else 0.0
    return num_same / n_gold


def retrieval_hit_at_k(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """Supporting-fact recall@k: fraction of gold titles present in the retrieved set.

    This is the "top" metric: did the retriever surface the paragraphs that
    actually contain the answer, within the top-k it returned?
    """
    if not gold_titles:
        return 0.0
    gold = set(gold_titles)
    retrieved = set(retrieved_titles)
    return len(gold & retrieved) / len(gold)


def answer_in_context(gold_answer: str, contexts: list[str]) -> float:
    """1.0 if the normalized gold answer appears verbatim in any retrieved chunk.

    A retrieval-quality signal that is robust to title/paragraph alignment:
    even if titles do not match, did we actually retrieve text containing the
    answer? (Always 0.0 for yes/no answers that normalize to a stopword-free
    token only when that token is present — handled by normalization.)
    """
    needle = normalize_answer(gold_answer)
    if not needle:
        return 0.0
    haystack = normalize_answer(" ".join(contexts))
    return float(needle in haystack)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_latency(latencies_ms: list[float]) -> dict[str, float]:
    """Mean / median / p95 of a list of latencies (ms)."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(latencies_ms)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean_ms": statistics.mean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_idx],
    }
