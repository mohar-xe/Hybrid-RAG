"""Entity and relationship extraction from text chunks.

Two interchangeable backends are available; choose one per call / CLI run:

* ``local``    — the Ollama-hosted fine-tuned model (``ExtractionSettings``),
                 via Ollama's native ``/api/chat`` endpoint.
* ``deepseek`` — a remote OpenAI-compatible endpoint (``NERSettings``),
                 defaulting to DeepSeek V4 Flash, authed with ``NER__API_KEY``.

``extract_entities(text, backend=...)`` dispatches between them. When ``backend``
is omitted it falls back to ``EXTRACTION__BACKEND`` (default: ``deepseek``).
If the primary backend fails, the other backend is tried as a fallback.
"""

import json
from collections.abc import Iterable

import httpx

from graph.schema import Triplet, VALID_RELATION_TYPES
from config.settings import ExtractionSettings, NERSettings
from constants.exceptions import ConfigurationError
from constants.logger import setup_logger
from models.fallback import with_fallback
from models.rate_limiter import GEMINI_LIMITER

LOGGER = setup_logger(__name__)

extraction = ExtractionSettings()  # local (Ollama) backend config
ner = NERSettings()  # remote (DeepSeek / OpenAI-compatible) backend config

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
        content = (
            content.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

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
                {
                    "role": "system",
                    "content": _build_system_prompt(
                        extraction.min_triplets, extraction.max_triplets
                    ),
                },
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
            {
                "role": "system",
                "content": _build_system_prompt(ner.min_triplets, ner.max_triplets),
            },
            {"role": "user", "content": f"Passage:\n{text}"},
        ],
        "temperature": ner.temperature,
        "max_tokens": 32768,
        "stream": False,
    }
    if ner.disable_thinking and "deepseek" in ner.model.lower():
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
    LOGGER.info(f"Extracted {len(triplets)} triplets (api/{ner.model}).")
    return triplets


# Selectable backends. Keys are the values accepted by `--extractor` / EXTRACTION__BACKEND.
_BACKENDS = {
    "local": extract_entities_local,
    "deepseek": extract_entities_api,
    "api": extract_entities_api,  # alias for "deepseek"
}


def _fallback_dispatcher(text: str, backend: str | None = None) -> list[Triplet]:
    """Run extraction with the configured primary backend, falling back to the other.

    On failure the primary backend is tried first; if it fails and fallback is
    enabled, the other backend is tried as a transparent fallback.
    """
    name = (backend or extraction.backend).lower()
    if name not in _BACKENDS:
        raise ConfigurationError(
            f"Unknown extraction backend: {backend!r}. Choose 'local' or 'deepseek'."
        )

    primary_fn = _BACKENDS[name]
    fallback_name = "local" if name in ("deepseek", "api") else "deepseek"
    fallback_fn = _BACKENDS.get(fallback_name)
    fallback_enabled = extraction.fallback_enabled and ner.fallback_enabled

    result, _tag = with_fallback(
        primary_fn,
        fallback_fn,
        "kg_extraction",
        fallback_enabled=fallback_enabled,
        primary_tag=name,
        fallback_tag=fallback_name,
        text=text,
    )
    return result


def extract_entities(text: str, backend: str | None = None) -> list[Triplet]:
    """Extract triplets using the selected backend with automatic fallback.

    Args:
        text: passage to extract triplets from.
        backend: ``"local"`` or ``"deepseek"``. When ``None``, falls back to
            ``EXTRACTION__BACKEND`` (default ``"deepseek"``).

    Raises:
        ConfigurationError: if ``backend`` is unknown, or the ``deepseek``
            backend is chosen without ``NER__API_KEY`` set.
    """
    return _fallback_dispatcher(text, backend)


def _resolve_concurrency(backend_name: str) -> int:
    """Per-backend default concurrency (NER__CONCURRENCY / EXTRACTION__CONCURRENCY)."""
    if backend_name in ("deepseek", "api"):
        return ner.concurrency
    return extraction.concurrency


def _extract_bundled_prompt(texts: list[str], start_idx: int) -> tuple[str, str]:
    """Build a system + user prompt for bundled extraction of multiple texts.

    Each text is separated by a ``---CHUNK N---`` marker. The model is instructed
    to return a JSON object mapping chunk numbers to triplet arrays.
    """
    sep = "\n\n---CHUNK %d---\n\n"
    parts = []
    for i, t in enumerate(texts):
        parts.append(f"Passage {start_idx + i}:\n{t}")
    combined = "\n\n".join(parts)

    system = _build_system_prompt(ner.min_triplets, ner.max_triplets)
    system += (
        "\n\nExtract triplets from EACH passage. Passages are numbered above."
        "\nOutput a JSON object where keys are passage indices (as strings: "
        '"0", "1", ...) and values are arrays of triplets for that passage. '
        "Output ONLY the JSON object, no prose, no markdown."
    )
    return system, combined


def _parse_bundled_response(
    content: str,
    batch_size: int,
    start_idx: int,
) -> list[list[Triplet]]:
    """Parse a bundled model response into per-chunk triplet lists.

    Expects ``{"0": [...], "1": [...], ...}`` or a flat array of triplets
    (non-bundled fallback). Uses raw JSON parsing (not ``_parse_json`` which
    wraps non-array values in a list).
    """
    content = content.strip()
    if content.startswith("```"):
        content = (
            content.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

    if not content:
        return [[] for _ in range(batch_size)]

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse bundled response JSON: %s", exc)
        return [[] for _ in range(batch_size)]

    # Dict: key=chunk_index str, value=list of triplet dicts
    if isinstance(raw, dict):
        result = [[] for _ in range(batch_size)]
        for key, items in raw.items():
            try:
                idx = int(key) - start_idx
            except (ValueError, TypeError):
                continue
            if 0 <= idx < batch_size and isinstance(items, list):
                result[idx] = validate_triplets(items)
        return result

    # Flat array: assign all triplets to the first chunk (non-bundled fallback)
    if isinstance(raw, list):
        triplets = validate_triplets(raw)
        result = [[] for _ in range(batch_size)]
        if triplets:
            result[0] = triplets
        return result

    LOGGER.warning(
        "Unexpected bundled response type %s; treating as empty.",
        type(raw).__name__,
    )
    return [[] for _ in range(batch_size)]


def _extract_entities_gemini_batch(
    texts: list[str],
) -> list[list[Triplet] | None]:
    """Extract triplets via Gemini API, bundling chunks per API call.

    Groups texts into batches of ``ner.batch_size`` (default 30), sends one
    API call per batch with a combined prompt, and rate-limits to 5 RPM.
    """
    if not texts:
        return []

    batch_size = ner.batch_size
    results: list[list[Triplet] | None] = [None] * len(texts)

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]

        GEMINI_LIMITER.acquire()

        system, combined = _extract_bundled_prompt(batch, batch_start)

        api_key = ner.api_key.get_secret_value()
        if not api_key:
            LOGGER.warning("NER__API_KEY not set; skipping bundled extraction.")
            for j in range(len(batch)):
                results[batch_start + j] = []
            continue

        payload = {
            "model": ner.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": combined},
            ],
            "temperature": ner.temperature,
            "max_tokens": 32768,
            "stream": False,
        }
        if ner.disable_thinking and "deepseek" in ner.model.lower():
            payload["thinking"] = {"type": "disabled"}

        try:
            response = httpx.post(
                f"{ner.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=ner.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            per_chunk = _parse_bundled_response(content, len(batch), batch_start)
            for j, triplets in enumerate(per_chunk):
                results[batch_start + j] = triplets
            LOGGER.info(
                "NER batch %d-%d: extracted %d triplets across %d chunks.",
                batch_start,
                batch_start + len(batch) - 1,
                sum(len(t) for t in per_chunk),
                len(batch),
            )
        except Exception as exc:
            LOGGER.warning(
                "Gemini batch extraction failed for batch %d: %s", batch_start, exc
            )
            for j in range(len(batch)):
                results[batch_start + j] = []

    return results


def extract_entities_batch(
    texts: list[str],
    backend: str | None = None,
    max_workers: int | None = None,
) -> list[list[Triplet] | None]:
    """Extract triplets for many texts.

    When the backend is ``"deepseek"``/``"api"`` and ``NER__BATCH_SIZE > 1``,
    texts are bundled and sent to Gemini (rate-limited, 5 RPM). Otherwise the
    original per-chunk concurrent path is used.

    Returns one entry per input text, in the **same order**:
    * ``list[Triplet]`` (possibly empty) on success,
    * ``None`` if that text's extraction raised (transient API/network error or
      bad response) — the caller can count/skip these without aborting the run.
    """
    if not texts:
        return []

    name = (backend or extraction.backend).lower()
    if name not in _BACKENDS:
        raise ConfigurationError(
            f"Unknown extraction backend: {backend!r}. Choose 'local' or 'deepseek'."
        )

    # Use the bundled Gemini path when batch_size > 1 and backend is API.
    if name in ("deepseek", "api") and ner.batch_size > 1:
        return _extract_entities_gemini_batch(texts)

    from concurrent.futures import ThreadPoolExecutor, as_completed

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
        content = (
            content.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
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
    return _dedupe_preserve(
        [item.strip() for item in data if isinstance(item, str) and item.strip()]
    )


def _extract_query_entities_api(query: str) -> list[str]:
    """Remote API query-entity extraction (DeepSeek / OpenAI-compatible)."""
    api_key = ner.api_key.get_secret_value()
    if not api_key:
        raise ConfigurationError(
            "NER API key is not set. Add NER__API_KEY to your .env to use the "
            "'deepseek' query-entity extraction backend."
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
        "max_tokens": 32768,
        "stream": False,
    }
    if ner.disable_thinking and "deepseek" in ner.model.lower():
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
    LOGGER.info(f"LLM query NER extracted {len(entities)} entities (api/{ner.model}).")
    return entities


def _extract_query_entities_ollama(query: str) -> list[str]:
    """Local Ollama query-entity extraction using the fine-tuned model."""
    system = (
        "You extract the named entities and salient noun-phrase concepts from a "
        "search question — the things a knowledge graph would index (people, "
        "organizations, places, works, systems, models, theories, events, "
        "methods). Return ONLY a JSON array of the entity surface strings exactly "
        'as a graph would store them (e.g. ["Albert Einstein", "Nobel Prize"]). '
        "No prose, no markdown, no objects, no duplicates."
    )
    response = httpx.post(
        f"{extraction.base_url.rstrip('/')}/api/chat",
        json={
            "model": extraction.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "options": {"temperature": extraction.temperature},
        },
        timeout=extraction.timeout,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]

    entities = _parse_string_array(content)
    LOGGER.info(
        f"LLM query NER extracted {len(entities)} entities (local/{extraction.model})."
    )
    return entities


def extract_query_entities_llm(query: str) -> list[str]:
    """LLM fallback NER for a query: return entity/concept surface strings.

    Tries the remote API first (``NER__*``, default DeepSeek), falling back to
    the local Ollama model (``EXTRACTION__*``) on failure.
    """
    result, _tag = with_fallback(
        _extract_query_entities_api,
        _extract_query_entities_ollama,
        "query_entities",
        fallback_enabled=ner.fallback_enabled and extraction.fallback_enabled,
        primary_tag="deepseek",
        fallback_tag="local",
        query=query,
    )
    return result


def extract_query_entities(query: str, *, use_llm_fallback: bool = True) -> list[str]:
    """Canonical query entity extractor used to seed graph retrieval.

    LLM-first, YAKE-fallback (opposite of the old default — deployment has
    a remote NER API but no YAKE model files). Falls back to YAKE keyphrases
    when the LLM call fails.

    Pass ``use_llm_fallback=False`` for callers that must stay cheap/offline
    (e.g. the router's complexity signal), restricting extraction to YAKE only.
    """
    if use_llm_fallback:
        try:
            return extract_query_entities_llm(query)
        except Exception as exc:
            LOGGER.warning(f"LLM query-entity NER failed, falling back to YAKE: {exc}")

    phrases = _dedupe_preserve(extract_keyphrases_yake(query))
    return phrases


# ---------------------------------------------------------------------------
# Batch query-entity extraction (eval harness, LLM-only)
# ---------------------------------------------------------------------------
# The eval harness extracts query entities for ALL eval questions up front, in
# ONE bundled Gemini call (``EXTRACTION__QUERY_ENTITY_MODEL``, default Pro
# Preview for quality). No YAKE — the user explicitly wants pure-LLM extraction
# here; a missing/corrupt batch row degrades to "no graph facts" for that query
# (the soft-boost path), never an error.


def extract_query_entities_batch(
    questions: list[str],
    *,
    model: str | None = None,
) -> list[list[str]]:
    """Extract entities for many questions in a single bundled LLM call.

    Pure-LLM (no YAKE). One ``GEMINI_LIMITER``-paced call for all questions;
    responses map question indices to entity arrays. Returns one list per
    question in input order; extraction failure for a question yields ``[]``
    (never raises), matching the soft-boost semantics of graph seeding.
    """
    if not questions:
        return []

    api_key = ner.api_key.get_secret_value()
    if not api_key:
        LOGGER.warning("NER__API_KEY not set; returning empty entities.")
        return [[] for _ in questions]

    resolved_model = model or extraction.query_entity_model or ner.model

    system = (
        "You extract the named entities and salient noun-phrase concepts from "
        "search questions — the things a knowledge graph would index (people, "
        "organizations, places, works, systems, models, theories, events, "
        "methods). For EACH numbered question below, return a JSON object "
        'mapping question indices (as strings: "0", "1", ...) to arrays of '
        "entity surface strings exactly as a graph would store them "
        '(e.g. {"0": ["Albert Einstein", "Nobel Prize"], "1": [...]}). '
        "No prose, no markdown, no duplicates, no objects other than the top-level map."
    )
    parts = [f"Q{i}: {q}" for i, q in enumerate(questions)]
    combined = "\n\n".join(parts)

    GEMINI_LIMITER.acquire()

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": combined},
        ],
        "temperature": ner.temperature,
        "max_tokens": 32768,
        "stream": False,
    }

    try:
        response = httpx.post(
            f"{ner.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=ner.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        LOGGER.warning("Bundled query-entity extraction failed: %s", exc)
        return [[] for _ in questions]

    content = (content or "").strip()
    if content.startswith("```"):
        content = (
            content.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

    results: list[list[str]] = [[] for _ in questions]
    if not content:
        return results

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Bundled query-entity response unparseable: %s", exc)
        return results

    if isinstance(raw, dict):
        for key, items in raw.items():
            try:
                idx = int(key)
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(questions) and isinstance(items, list):
                results[idx] = _dedupe_preserve(
                    [
                        item.strip()
                        for item in items
                        if isinstance(item, str) and item.strip()
                    ]
                )
    elif isinstance(raw, list) and all(isinstance(i, str) for i in raw):
        # Degenerate flat-array response: assign to every question (better than
        # nothing for the soft-boost path) — mirrors _parse_bundled_response.
        flat = _dedupe_preserve([i.strip() for i in raw if i.strip()])
        for idx in range(len(questions)):
            results[idx] = flat

    found = sum(1 for r in results if r)
    LOGGER.info(
        "Bundled query NER (%s): entities for %d/%d questions.",
        resolved_model,
        found,
        len(questions),
    )
    return results
