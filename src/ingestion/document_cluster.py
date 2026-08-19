"""Document-level metadata extraction and clustering (replaces K-Means/medoids).

Extracts rich metadata per document via Gemini 3.6 Flash (single request,
truncated to 500K chars) or falls back to spaCy heuristics when the API is
unavailable or the document exceeds METADATA__MAX_DOC_CHARS (1M).
"""

import hashlib
import json
import re
import uuid

import httpx
import psycopg

from config.settings import get_settings
from constants.logger import setup_logger
from constants.exceptions import ConfigurationError
from embeddings.embedder import embedder
from models.fallback import with_fallback
from cache import metadata_cache

LOGGER = setup_logger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """Extract document metadata from the full text below. Return a JSON object with these fields:
{
  "title": "descriptive title extracted from content (never the filename)",
  "summary": "2-4 sentence abstractive summary of the document",
  "synthetic_questions": ["question that this document can answer", ...],
  "doc_type": "document type like contract, report, manual, transcript, academic, article, note, email, or other",
  "topic_tags": ["key-topic-keywords", ...],
  "entities": [{"name": "EntityName", "type": "PERSON|ORG|GPE|PRODUCT|EVENT|WORK_OF_ART|LAW|TECHNOLOGY"}],
  "content_date": "ISO date string or null if no temporal scope can be determined",
  "version_info": "version string like 'v2.1', 'draft 3', or null"
}
Output ONLY the JSON. No prose, no markdown, no explanation."""

_TRUNCATION_LIMIT = 500_000


def _call_metadata_api(text: str) -> dict:
    """Send a single request to Gemini 3.6 Flash for metadata extraction."""
    api_key = (
        settings.metadata.api_key.get_secret_value()
        or settings.ner.api_key.get_secret_value()
    )
    if not api_key:
        raise ConfigurationError(
            "Metadata extraction: no API key configured. Set METADATA__API_KEY "
            "or NER__API_KEY in .env, or use METADATA__BACKEND=local for spaCy only."
        )
    payload = {
        "model": settings.metadata.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
        "stream": False,
    }
    response = httpx.post(
        f"{settings.metadata.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=settings.metadata.timeout,
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
    return json.loads(content)


def _extract_metadata_local(text: str) -> dict:
    """spaCy-only heuristic fallback — no API call.

    Returns the same schema as the Gemini path but with lower-quality signals
    (extractive summary, no synthetic questions).
    """
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        LOGGER.error("spacy not installed. Install it: uv sync --extra local")
        raise ConfigurationError(
            "spacy not installed. Install it: uv sync --extra local"
        )
    except OSError:
        LOGGER.error(
            "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
        )
        raise ConfigurationError(
            "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
        )

    # Process first 100K chars for speed (enough for title/entities/summary)
    sample = text[:100_000]
    doc = nlp(sample)

    # Title: first short non-empty line that looks like a heading
    title = ""
    for line in sample.split("\n"):
        line = line.strip()
        if line and len(line) < 200 and not line.endswith((".", "?", "!")):
            title = line
            break

    # Summary: lead 3 sentences via spaCy sentencizer
    sents = list(doc.sents)
    summary = " ".join(str(s) for s in sents[:3]) if sents else sample[:500]

    # Entities
    entities = []
    seen_ents: set[str] = set()
    for ent in doc.ents:
        key = f"{ent.text}|{ent.label_}"
        if key not in seen_ents and ent.label_ in {
            "PERSON",
            "ORG",
            "GPE",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
        }:
            seen_ents.add(key)
            entities.append({"name": ent.text, "type": ent.label_})

    # Topic tags: noun chunks + frequent entities
    topic_tags: list[str] = []
    seen_tags: set[str] = set()
    for chunk in doc.noun_chunks:
        text_lower = chunk.text.lower().strip()
        if text_lower and len(text_lower) > 2 and text_lower not in seen_tags:
            seen_tags.add(text_lower)
            topic_tags.append(chunk.text)
        if len(topic_tags) >= 20:
            break

    # Content date: first DATE entity or date regex
    content_date = None
    for ent in doc.ents:
        if ent.label_ == "DATE":
            content_date = ent.text
            break
    if not content_date:
        date_match = re.search(
            r"\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text[:2000]
        )
        if date_match:
            content_date = date_match.group()

    # Version info: regex scan of first 2K chars
    version_info = None
    head = text[:2000]
    version_patterns = [
        r"\bv(\d+(?:\.\d+)+)\b",
        r"\bversion\s+(\d+(?:\.\d+)*)\b",
        r"\bdraft\s+(\d+)\b",
        r"\brevision\s+(\d+)\b",
    ]
    for pat in version_patterns:
        m = re.search(pat, head, re.IGNORECASE)
        if m:
            version_info = m.group(0)
            break
    # Doc type from source_type hint — the caller passes text, but we infer
    doc_type = "document"

    return {
        "title": title or "Untitled",
        "summary": summary,
        "synthetic_questions": [],
        "doc_type": doc_type,
        "topic_tags": topic_tags[:15],
        "entities": entities[:30],
        "content_date": content_date,
        "version_info": version_info,
    }


def _apply_filename_version(metadata: dict, source_id: str) -> None:
    """Lowest-priority version detection: filename patterns like _v2, -draft."""
    if metadata.get("version_info") or not source_id:
        return
    for pat in [r"_v(\d+(?:\.\d+)*)", r"[-.](\d+)\.\w+$"]:
        m = re.search(pat, source_id)
        if m:
            metadata["version_info"] = m.group(0).lstrip("._-")
            break


def extract_document_metadata(text: str, source_id: str = "") -> dict:
    """Single-request metadata extraction via Gemini 3.6 Flash.

    Size strategy:
      - len(text) <= 500K chars → send full text
      - 500K < len(text) <= MAX_DOC_CHARS → truncate to first 500K
      - len(text) > MAX_DOC_CHARS → skip Gemini, return spaCy local fallback

    On Gemini failure (timeout, auth, rate limit), falls back to
    ``_extract_metadata_local`` via ``with_fallback``.
    """
    _cache_key = hashlib.md5(text[:500].encode("utf-8")).hexdigest() + "|" + source_id
    cached = metadata_cache.get(_cache_key)
    if cached is not None:
        LOGGER.debug("Metadata cache hit for source=%s", source_id)
        return cached

    if len(text) > settings.metadata.max_doc_chars:
        LOGGER.info(
            "Doc exceeds %d chars (%d), using local spaCy fallback.",
            settings.metadata.max_doc_chars,
            len(text),
        )
        result = _extract_metadata_local(text)
        _apply_filename_version(result, source_id)
        metadata_cache.set(_cache_key, result)
        return result

    api_text = text[:_TRUNCATION_LIMIT]

    def primary_fn(text_=api_text):
        return _call_metadata_api(text_)

    def fallback_fn(text_=api_text):
        return _extract_metadata_local(text_)

    try:
        result, tag = with_fallback(
            primary_fn,
            fallback_fn,
            "metadata_extraction",
            fallback_enabled=settings.metadata.fallback_enabled,
            primary_tag="api" if settings.metadata.backend == "api" else "local",
            fallback_tag="local",
        )
        LOGGER.info("Metadata extraction completed via %s.", tag)
    except Exception as exc:
        LOGGER.error("Metadata extraction failed (all backends): %s", exc)
        result = {
            "title": "Untitled",
            "summary": "",
            "synthetic_questions": [],
            "doc_type": "",
            "topic_tags": [],
            "entities": [],
            "content_date": None,
            "version_info": None,
        }

    _apply_filename_version(result, source_id)
    metadata_cache.set(_cache_key, result)
    return result


def create_document_cluster(
    doc_id: str,
    source_id: str,
    source_type: str,
    metadata: dict,
    text: str,
    *,
    supersedes_doc_id: str | None = None,
    is_versioned: bool = False,
    version_label: str | None = None,
    summary_embedding: list[float] | None = None,
) -> None:
    """Embed summary + questions via embedder, write document_clusters row
    and document_questions rows to pgvector.

    If metadata extraction produced partial/no data, writes a minimal row so
    every document has a cluster entry (data-quality gradient).

    Args:
        supersedes_doc_id: doc_id this version replaces (set via CLI/API flag).
        is_versioned: whether this document is part of a version chain.
        version_label: display label (e.g. "v2", "draft 3"). Falls back to
                       metadata['version_info'] when not explicitly provided.
        summary_embedding: precomputed embedding for ``summary`` (batched callers
                           embed many summaries in one API request and pass the
                           vectors in, avoiding one request per document).
    """
    effective_version = version_label or metadata.get("version_info")
    summary = metadata.get("summary") or text[:500]
    if summary_embedding is None:
        summary_embedding = embedder([summary])[0]

    questions = metadata.get("synthetic_questions") or []
    question_embeddings = embedder(questions) if questions else []

    with psycopg.connect(settings.database.conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO document_clusters
                   (doc_id, source_id, source_type, title, summary,
                    summary_embedding, doc_type, topic_tags, content_date,
                    supersedes_doc_id, is_versioned, version_label,
                    entities, metadata_json)
                   VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s,
                           %s, %s, %s, %s, %s)
                   ON CONFLICT (doc_id) DO NOTHING""",
                (
                    doc_id,
                    source_id,
                    source_type,
                    metadata.get("title", "Untitled"),
                    summary,
                    _vec_literal(summary_embedding),
                    metadata.get("doc_type", ""),
                    metadata.get("topic_tags", []),
                    metadata.get("content_date"),
                    supersedes_doc_id,
                    is_versioned or bool(supersedes_doc_id),
                    effective_version,
                    json.dumps(metadata.get("entities", [])),
                    json.dumps(metadata),
                ),
            )

            for q, emb in zip(questions, question_embeddings):
                cur.execute(
                    """INSERT INTO document_questions (id, doc_id, question, embedding)
                       VALUES (%s, %s, %s, %s::vector)
                       ON CONFLICT (id) DO NOTHING""",
                    (str(uuid.uuid4()), doc_id, q, _vec_literal(emb)),
                )

        conn.commit()

    LOGGER.info(
        "Created document cluster %s (title=%r, %d questions, version=%s).",
        doc_id,
        metadata.get("title", "Untitled"),
        len(questions),
        effective_version or "none",
    )


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"
