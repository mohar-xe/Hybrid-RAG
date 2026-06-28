"""Entity and relationship extraction from text chunks.

Two interchangeable backends are available; choose one per call / CLI run:

* ``local``    — the Ollama-hosted fine-tuned model (``ExtractionSettings``),
                 via Ollama's native ``/api/chat`` endpoint.
* ``deepseek`` — a remote OpenAI-compatible endpoint (``NERSettings``),
                 defaulting to DeepSeek V4 Flash, authed with ``NER__API_KEY``.

``extract_entities(text, backend=...)`` dispatches between them. When ``backend``
is omitted it falls back to ``EXTRACTION__BACKEND`` (default: ``deepseek``).
"""

import json
from collections.abc import Iterable

import httpx

from graph.schema import Triplet, VALID_RELATION_TYPES
from config.settings import ExtractionSettings, NERSettings
from constants.exceptions import ConfigurationError
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)

extraction = ExtractionSettings()   # local (Ollama) backend config
ner = NERSettings()                 # remote (DeepSeek / OpenAI-compatible) backend config

_RELATION_LIST = ", ".join(sorted(VALID_RELATION_TYPES))

def _build_system_prompt(min_triplets: int, max_triplets: int) -> str:
    """Build the extraction system prompt for the given triplet-count bounds."""
    return f"""You are a knowledge graph extraction engine.

Extract relation triplets from the passage strictly following this JSON schema:
{{
  "source": {{"title": string, "type": "entity" | "concept"}},
  "relation": {{"type": string, "weight": float}},
  "target": {{"title": string, "type": "entity" | "concept"}}
}}

Rules:
- source.type / target.type must be exactly "entity" or "concept"
  entity = named grounded thing (person, org, place, tool, model, dataset, system)
  concept = abstract or categorical (theory, method, process, property, domain, metric)
- relation.type must be one of:
  {_RELATION_LIST}
- weight: 0.8–1.0 if explicit and central | 0.5–0.8 if implicit but clear | 0.1–0.4 if inferred
- Extract {min_triplets} to {max_triplets} triplets per passage
- Output a JSON array only. No prose, no markdown, no explanation."""


def _parse_json(content: str) -> list[dict]:
    """Extract the JSON array of raw triplet dicts from a model response.

    Handles ```/```json code fences and a single (non-array) object. Does NOT
    validate against the schema — that is ``validate_triplets``'s job.
    """
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    if not content:
        # Empty model response -> no triplets (not an error).
        LOGGER.warning("Empty extraction response; no triplets parsed.")
        return []

    data = json.loads(content)
    if not isinstance(data, list):
        data = [data]
    return data


def validate_triplets(items: Iterable[dict | Triplet]) -> list[Triplet]:
    """Schema-check triplets, returning only the schema-valid ones.

    Idempotent: an item that is already a validated ``Triplet`` instance is
    passed through unchanged (the check is skipped for it). A raw ``dict`` is
    validated by constructing a ``Triplet``; items that violate the schema —
    unknown ``relation.type``, a node ``type`` other than ``entity``/``concept``,
    ``weight`` outside ``[0, 1]``, or missing/extra fields — are dropped.

    Used by both the Ollama (local) and DeepSeek (api) extraction backends.
    """
    validated: list[Triplet] = []
    skipped = invalid = 0
    for item in items:
        if isinstance(item, Triplet):
            validated.append(item)  # already validated -> skip re-checking
            skipped += 1
            continue
        try:
            validated.append(Triplet(**item))
        except Exception as exc:
            invalid += 1
            LOGGER.debug(f"Dropped triplet failing schema check: {exc}")

    if invalid:
        LOGGER.warning(f"Schema check dropped {invalid} invalid triplet(s).")
    if skipped:
        LOGGER.debug(f"Schema check: {skipped} already-validated triplet(s) skipped.")
    return validated


def extract_entities_local(text: str) -> list[Triplet]:
    """Extract triplets using the local Ollama fine-tuned model (native /api/chat)."""
    options: dict = {"temperature": extraction.temperature}
    if extraction.num_ctx is not None:
        options["num_ctx"] = extraction.num_ctx

    response = httpx.post(
        f"{extraction.base_url.rstrip('/')}/api/chat",
        json={
            "model": extraction.model,
            "messages": [
                {"role": "system", "content": _build_system_prompt(extraction.min_triplets, extraction.max_triplets)},
                {"role": "user", "content": f"Passage:\n{text}"},
            ],
            "stream": False,
            "options": options,
        },
        timeout=extraction.timeout,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]

    triplets = validate_triplets(_parse_json(content))
    LOGGER.info(f"Extracted {len(triplets)} triplets (local/{extraction.model}).")
    return triplets


def extract_entities_api(text: str) -> list[Triplet]:
    """Extract triplets via a remote OpenAI-compatible endpoint (default: DeepSeek V4 Flash)."""
    api_key = ner.api_key.get_secret_value()
    if not api_key:
        raise ConfigurationError(
            "NER API key is not set. Add NER__API_KEY to your .env to use the "
            "'deepseek' extraction backend, or select the local backend "
            "(--extractor local / EXTRACTION__BACKEND=local)."
        )

    payload = {
        "model": ner.model,
        "messages": [
            {"role": "system", "content": _build_system_prompt(ner.min_triplets, ner.max_triplets)},
            {"role": "user", "content": f"Passage:\n{text}"},
        ],
        "temperature": ner.temperature,
        "max_tokens": 1024,
        "stream": False,
    }
    if ner.disable_thinking:
        # DeepSeek hybrid models default to reasoning; disabling it keeps the
        # token budget for the actual JSON answer and cuts latency.
        payload["thinking"] = {"type": "disabled"}

    response = httpx.post(
        f"{ner.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=ner.timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    triplets = validate_triplets(_parse_json(content))
    LOGGER.info(f"Extracted {len(triplets)} triplets (deepseek/{ner.model}).")
    return triplets


# Selectable backends. Keys are the values accepted by `--extractor` / EXTRACTION__BACKEND.
_BACKENDS = {
    "local": extract_entities_local,
    "deepseek": extract_entities_api,
    "api": extract_entities_api,  # alias for "deepseek"
}


def extract_entities(text: str, backend: str | None = None) -> list[Triplet]:
    """Extract triplets using the selected backend.

    Args:
        text: passage to extract triplets from.
        backend: ``"local"`` or ``"deepseek"``. When ``None``, falls back to
            ``EXTRACTION__BACKEND`` (default ``"deepseek"``).

    Raises:
        ConfigurationError: if ``backend`` is unknown, or the ``deepseek``
            backend is chosen without ``NER__API_KEY`` set.
    """
    name = (backend or extraction.backend).lower()
    fn = _BACKENDS.get(name)
    if fn is None:
        raise ConfigurationError(
            f"Unknown extraction backend: {backend!r}. Choose 'local' or 'deepseek'."
        )
    return fn(text)


def _resolve_concurrency(backend_name: str) -> int:
    """Per-backend default concurrency (NER__CONCURRENCY / EXTRACTION__CONCURRENCY)."""
    if backend_name in ("deepseek", "api"):
        return ner.concurrency
    return extraction.concurrency


def extract_entities_batch(
    texts: list[str],
    backend: str | None = None,
    max_workers: int | None = None,
) -> list[list[Triplet] | None]:
    """Extract triplets for many texts concurrently (the calls are I/O-bound).

    Returns one entry per input text, in the **same order**:
    * ``list[Triplet]`` (possibly empty) on success,
    * ``None`` if that text's extraction raised (transient API/network error or
      bad response) — the caller can count/skip these without aborting the run.

    Concurrency defaults to the backend's configured value
    (``NER__CONCURRENCY`` for ``deepseek``, ``EXTRACTION__CONCURRENCY`` for
    ``local``). Kùzu writes are NOT done here — do them serially in the caller,
    since a Kùzu connection is not thread-safe.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not texts:
        return []

    name = (backend or extraction.backend).lower()
    if name not in _BACKENDS:
        raise ConfigurationError(
            f"Unknown extraction backend: {backend!r}. Choose 'local' or 'deepseek'."
        )
    workers = max_workers or _resolve_concurrency(name)
    workers = max(1, min(workers, len(texts)))

    results: list[list[Triplet] | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(extract_entities, text, backend): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                LOGGER.warning(f"Extraction failed for item {i}: {exc}")
                results[i] = None
    return results


# Backward-compatible alias: older callers / notebooks import this name.
extract_entities_llm = extract_entities_local


# ---------------------------------------------------------------------------
# Query-time entity extraction (hybrid: YAKE keyphrases -> LLM fallback)
# ---------------------------------------------------------------------------
# The retrieval paths (CLI `ask`, FastAPI `/query`, the Gradio demo, and the
# forced-search module) need the *entities mentioned in a question* to seed the
# Kùzu graph lookup (`get_entity_context`). This is the single canonical home
# for that, replacing the old crude "capitalized whitespace tokens" heuristic
# that was duplicated at every call site.
#
# Strategy (cheap-first): run YAKE — a fast, local, unsupervised keyphrase
# extractor already used during ingestion — and use its keyphrases as graph
# seeds. Only when YAKE finds nothing (e.g. a pronoun-only question like "tell
# me about it") do we pay for an LLM call to recover entities. YAKE preserves
# the original casing of phrases ("Albert Einstein", "Nobel Prize"), which
# matters because `get_entity_context` matches `Entity.name` *exactly*.

# Query-tuned YAKE knobs, deliberately more permissive than the chunker's
# indexing config: a high dedup limit keeps multi-word phrases *and* their
# component tokens ("Albert Einstein" *and* "Einstein") so more candidate seeds
# get a chance to match a stored Entity name. Surplus non-matching seeds are
# harmless (they simply find no fact); missing a real entity is not.
_QUERY_YAKE_NGRAM = 3
_QUERY_YAKE_TOP = 10
_QUERY_YAKE_DEDUP = 0.9


def _dedupe_preserve(items: list[str]) -> list[str]:
    """Order-preserving de-duplication of a string list."""
    return list(dict.fromkeys(items))


def extract_keyphrases_yake(
    text: str,
    *,
    n: int = _QUERY_YAKE_NGRAM,
    top: int = _QUERY_YAKE_TOP,
    dedup_lim: float = _QUERY_YAKE_DEDUP,
) -> list[str]:
    """Return YAKE keyphrases for ``text`` (original casing preserved).

    Resilient by design: any failure (missing/broken YAKE, empty text) yields an
    empty list rather than raising, because this runs in the online query hot
    path. An empty return is also the signal the hybrid extractor uses to fall
    back to the LLM.
    """
    if not text or not text.strip():
        return []
    try:
        import yake

        extractor = yake.KeywordExtractor(lan="en", n=n, dedup_lim=dedup_lim, top=top)
        return [kw for kw, _score in extractor.extract_keywords(text)]
    except Exception as exc:
        LOGGER.warning(f"YAKE keyphrase extraction failed: {exc}")
        return []


def _parse_string_array(content: str) -> list[str]:
    """Parse a model response into a de-duplicated list of non-empty strings.

    Tolerates ``` / ```json fences and an accidental object wrapper
    (``{"entities": [...]}``); silently yields ``[]`` for an empty/garbled
    response (the caller treats "no entities" as "skip the graph", never error).
    """
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not content:
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        LOGGER.warning("LLM query NER returned non-JSON; ignoring.")
        return []
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    if not isinstance(data, list):
        return []
    return _dedupe_preserve([item.strip() for item in data if isinstance(item, str) and item.strip()])


def extract_query_entities_llm(query: str) -> list[str]:
    """LLM fallback NER for a query: return entity/concept surface strings.

    Uses the remote OpenAI-compatible endpoint (``NER__*``, default DeepSeek) —
    a general model with a query-focused NER prompt, which recovers entities
    better than keyphrase statistics when YAKE comes up empty. Raises
    ``ConfigurationError`` if the API key is unset (the hybrid wrapper catches
    it and degrades to no seeds).
    """
    api_key = ner.api_key.get_secret_value()
    if not api_key:
        raise ConfigurationError(
            "NER API key is not set. Add NER__API_KEY to your .env to use the "
            "LLM query-entity fallback (the 'deepseek' backend)."
        )

    system = (
        "You extract the named entities and salient noun-phrase concepts from a "
        "search question — the things a knowledge graph would index (people, "
        "organizations, places, works, systems, models, theories, events, "
        "methods). Return ONLY a JSON array of the entity surface strings exactly "
        'as a graph would store them (e.g. ["Albert Einstein", "Nobel Prize"]). '
        "No prose, no markdown, no objects, no duplicates."
    )
    payload = {
        "model": ner.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "temperature": ner.temperature,
        "max_tokens": 256,
        "stream": False,
    }
    if ner.disable_thinking:
        # DeepSeek hybrid models default to reasoning; disabling it keeps the
        # token budget for the JSON answer and cuts latency.
        payload["thinking"] = {"type": "disabled"}

    response = httpx.post(
        f"{ner.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=ner.timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    entities = _parse_string_array(content)
    LOGGER.info(f"LLM query NER extracted {len(entities)} entities (deepseek/{ner.model}).")
    return entities


def extract_query_entities(query: str, *, use_llm_fallback: bool = True) -> list[str]:
    """Canonical query entity extractor used to seed graph retrieval.

    Hybrid, cheap-first:
      1. YAKE keyphrases — fast, local, no API cost. If it returns anything,
         those are the seeds ("if YAKE produces entities, good").
      2. Otherwise (YAKE empty) fall back to the LLM NER for better recall.

    The LLM step is best-effort: a missing API key or a transient error degrades
    to an empty list (the query still answers, just without graph facts). Pass
    ``use_llm_fallback=False`` for callers that must stay cheap/offline (e.g. the
    router's complexity signal), restricting extraction to YAKE only.
    """
    phrases = _dedupe_preserve(extract_keyphrases_yake(query))
    if phrases:
        return phrases
    if not use_llm_fallback:
        return []
    try:
        return extract_query_entities_llm(query)
    except Exception as exc:
        LOGGER.warning(f"LLM query-entity fallback unavailable: {exc}")
        return []
