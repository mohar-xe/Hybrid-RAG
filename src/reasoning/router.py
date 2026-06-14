"""Adaptive retrieval router — classify query complexity, select strategy."""

from enum import Enum

from constants.logger import setup_logger

LOGGER = setup_logger(__name__)


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


_COMPARISON_WORDS = {"compare", "contrast", "difference", "versus", "vs"}
_MULTI_HOP_WORDS = {"relate", "relationship", "connection", "influence", "impact"}
_BROAD_WORDS = {"summarize", "overview", "main themes", "explain all"}


def classify_query(query: str) -> QueryComplexity:
    q_lower = query.lower().strip()
    words = q_lower.split()
    word_count = len(words)
    entity_count = sum(1 for w in query.split() if w[0].isupper() and len(w) > 1)

    if word_count < 8 and any(q_lower.startswith(p) for p in ("what is", "who is", "when did", "where is", "define")):
        return QueryComplexity.SIMPLE

    if any(phrase in q_lower for phrase in _BROAD_WORDS):
        return QueryComplexity.COMPLEX

    if entity_count >= 3 or any(w in q_lower for w in _MULTI_HOP_WORDS):
        return QueryComplexity.COMPLEX

    if any(w in words for w in _COMPARISON_WORDS) or word_count > 12:
        return QueryComplexity.MODERATE

    return QueryComplexity.MODERATE


def route_retrieval(query: str) -> dict:
    complexity = classify_query(query)

    strategies = {
        QueryComplexity.SIMPLE: {
            "complexity": "simple",
            "use_vector": True,
            "use_bm25": False,
            "use_graph": False,
            "use_reranker": False,
            "top_k": 5,
        },
        QueryComplexity.MODERATE: {
            "complexity": "moderate",
            "use_vector": True,
            "use_bm25": True,
            "use_graph": False,
            "use_reranker": True,
            "top_k": 10,
        },
        QueryComplexity.COMPLEX: {
            "complexity": "complex",
            "use_vector": True,
            "use_bm25": True,
            "use_graph": True,
            "use_reranker": True,
            "top_k": 20,
        },
    }

    strategy = strategies[complexity]
    LOGGER.info(f"Query routed as '{complexity.value}': {query[:60]}...")
    return strategy