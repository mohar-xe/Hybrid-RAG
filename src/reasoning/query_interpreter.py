"""Query understanding: translate natural language into semantic_query + structured filters.

Uses Gemini 2.0 Flash-Lite for cheap/fast interpretation at query time.
Falls back to heuristic parsing when the API is unavailable.
"""

import hashlib
import json
import re

import httpx

from config.settings import get_settings
from constants.exceptions import ConfigurationError
from constants.logger import setup_logger
from cache import query_cache

LOGGER = setup_logger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """You are a query interpreter for a document retrieval system.
Given a user question, extract:
1. semantic_query: the core search terms (remove filter-like words like "latest", "recent")
2. filters: structured filters to narrow the search

Return ONLY valid JSON with this schema:
{
  "semantic_query": "cleaned search terms",
  "filters": {
    "doc_type": string or null,
    "is_latest": boolean or null,
    "date_after": "YYYY-MM-DD" or null,
    "entities": ["EntityName", ...]
  }
}

Rules:
- is_latest=true when the question asks for "latest", "most recent", "newest", "current"
- doc_type should infer from context: contract, report, manual, transcript, etc.
- date_after for temporal questions like "since 2023", "after March"
- entities: named people, organizations, products mentioned
- semantic_query: remove filter words, keep the core information need
Output ONLY the JSON. No prose."""


def interpret_query(question: str) -> dict:
    """Translate a user question into (semantic_query, filters).

    Returns a dict with keys ``semantic_query`` and ``filters``.
    Falls back to heuristic parsing on API failure.
    """
    _cache_key = hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()
    cached = query_cache.get(_cache_key)
    if cached is not None:
        LOGGER.debug("Query interpretation cache hit.")
        return cached

    api_key = (
        settings.query.api_key.get_secret_value()
        or settings.metadata.api_key.get_secret_value()
        or settings.ner.api_key.get_secret_value()
    )

    if api_key:
        try:
            payload = {
                "model": settings.query.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "temperature": settings.query.temperature,
                "max_tokens": 1024,
                "stream": False,
            }
            response = httpx.post(
                f"{settings.query.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=settings.query.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = (
                    content.removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
            result = json.loads(content)
            query_cache.set(_cache_key, result)
            return result
        except Exception as exc:
            LOGGER.warning("Query interpretation via API failed: %s", exc)

    result = _heuristic_interpret(question)
    query_cache.set(_cache_key, result)
    return result


def _heuristic_interpret(question: str) -> dict:
    """Cheap rule-based fallback when the API is unavailable."""
    q_lower = question.lower().strip()
    filters: dict = {
        "doc_type": None,
        "is_latest": None,
        "date_after": None,
        "entities": [],
    }

    if any(w in q_lower for w in ("latest", "newest", "most recent", "current")):
        filters["is_latest"] = True

    date_match = re.search(
        r"(?:since|after|from)\s+(20\d{2})(?:[-/](0[1-9]|1[0-2])(?:[-/](0[1-9]|[12]\d|3[01]))?)?",
        q_lower,
    )
    if date_match:
        year = date_match.group(1)
        month = date_match.group(2) or "01"
        day = date_match.group(3) or "01"
        filters["date_after"] = f"{year}-{month}-{day}"

    # Crude entity extraction: capitalized multi-word spans
    words = question.split()
    for i, w in enumerate(words):
        if w and w[0].isupper() and i > 0 and words[i - 1][0].islower():
            continue
        if (
            w
            and w[0].isupper()
            and w.lower() not in ("what", "who", "why", "when", "where", "how", "which")
        ):
            filters["entities"].append(w.strip(",:;.!?"))

    # Clean semantic query: remove filter trigger words
    semantic = q_lower
    for w in ("latest ", "most recent ", "newest ", "current "):
        if semantic.startswith(w):
            semantic = semantic[len(w) :]
            break

    return {
        "semantic_query": semantic,
        "filters": filters,
    }
