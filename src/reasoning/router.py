"""Adaptive retrieval router — classify query complexity, route to query interpreter."""

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
    entity_count = _named_entity_count(query)

    if word_count < 8 and any(
        q_lower.startswith(p)
        for p in ("what is", "who is", "when did", "where is", "define")
    ):
        return QueryComplexity.SIMPLE

    if any(phrase in q_lower for phrase in _BROAD_WORDS):
        return QueryComplexity.COMPLEX

    if entity_count >= 2 or any(w in q_lower for w in _MULTI_HOP_WORDS):
        return QueryComplexity.COMPLEX

    if any(w in words for w in _COMPARISON_WORDS) or word_count > 12:
        return QueryComplexity.MODERATE

    return QueryComplexity.MODERATE


def route_retrieval(query: str) -> dict:
    """Decouple complexity from hard-coded retrieval strategy.

    Returns the query complexity. Callers should route to query_interpreter
    for interpretation, then use ``document_routed_search`` with the
    interpreted filters — the old per-complexity bool flags are removed.
    """
    complexity = classify_query(query)
    LOGGER.info(f"Query routed as '{complexity.value}': {query[:60]}...")
    return {
        "complexity": complexity.value,
        "use_graph": complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX),
        "top_k": 5
        if complexity == QueryComplexity.SIMPLE
        else 10
        if complexity == QueryComplexity.MODERATE
        else 20,
    }
