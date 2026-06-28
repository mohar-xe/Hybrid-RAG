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


def _named_entity_count(query: str) -> int:
    """Approximate how many distinct named entities a query mentions.

    Uses YAKE keyphrases (the same extractor that seeds graph retrieval) instead
    of the old "every capitalized whitespace token" heuristic, then counts the
    distinct capitalized tokens inside those phrases — excluding the query's
    first word so a leading "What"/"Who"/"Compare" is not miscounted as an
    entity. YAKE-only: this runs on every query, so it must stay cheap and never
    make a network call (no LLM fallback here).
    """
    from graph.entity_extraction import extract_keyphrases_yake

    parts = query.split()
    first = parts[0].lower() if parts else ""
    tokens: set[str] = set()
    for phrase in extract_keyphrases_yake(query):
        for tok in phrase.split():
            if len(tok) > 1 and tok[0].isupper() and tok.lower() != first:
                tokens.add(tok)
    return len(tokens)


def classify_query(query: str) -> QueryComplexity:
    q_lower = query.lower().strip()
    words = q_lower.split()
    word_count = len(words)
    # Entity richness via YAKE keyphrases (replaces the old capitalized-token
    # split). Threshold is 2 (not 3) because the count no longer includes the
    # inflating sentence-initial capitalized word.
    entity_count = _named_entity_count(query)

    if word_count < 8 and any(q_lower.startswith(p) for p in ("what is", "who is", "when did", "where is", "define")):
        return QueryComplexity.SIMPLE

    if any(phrase in q_lower for phrase in _BROAD_WORDS):
        return QueryComplexity.COMPLEX

    if entity_count >= 2 or any(w in q_lower for w in _MULTI_HOP_WORDS):
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