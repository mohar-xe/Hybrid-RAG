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

    response = httpx.post(
        f"{ner.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": ner.model,
            "messages": [
                {"role": "system", "content": _build_system_prompt(ner.min_triplets, ner.max_triplets)},
                {"role": "user", "content": f"Passage:\n{text}"},
            ],
            "temperature": ner.temperature,
            "max_tokens": 1024,
            "stream": False,
        },
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


# Backward-compatible alias: older callers / notebooks import this name.
extract_entities_llm = extract_entities_local
