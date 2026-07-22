"""Recursive text chunking (Stage 1).

Splits normalized text into bounded ~300-token chunks *before* embedding, so no
single input can exceed the embedding model's context window (the previous
paragraph-embed-then-cluster approach could feed 25k-token blobs to Ollama and
400). Token counts are approximated from characters (~4 chars/token, see
``CHARS_PER_TOKEN``) — no real tokenizer dependency.

Algorithm (RecursiveCharacterTextSplitter style):
1. Recursively split on the highest-priority separator that appears
   (``["\\n\\n", "\\n", ". ", " "]``), descending to the next separator for any
   piece still larger than the target, and hard-splitting on characters as a
   last resort. This yields atomic pieces each <= the char budget.
2. Greedily merge atomic pieces back up to ``CHUNK_CHARS``, carrying an
   ``OVERLAP_CHARS`` tail of trailing pieces into the next chunk.
3. Embed every chunk in one batched pass (sizes are now bounded, so this is
   safe) and L2-normalize for cosine search.

Chunks are stored/searched under cosine distance (``vector_cosine_ops``).
"""

from dataclasses import dataclass
from typing import Literal
import re
import uuid

import numpy as np

from constants.logger import setup_logger
from constants.exceptions import ConfigurationError
from ingestion.normalize import split_into_paragraphs  # noqa: F401  (kept for callers/tests)
from embeddings.embedder import embedder
from ingestion.chunk_schema import Chunk

LOGGER = setup_logger(__name__)

# Char-based token approximation: ~4 chars per token for English text.
CHARS_PER_TOKEN = 4
CHUNK_TOKENS = 300
OVERLAP_TOKENS = 50
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN  # ~1200 chars
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~200 chars

# Separator priority: paragraph -> line -> sentence -> word -> character.
SEPARATORS = ["\n\n", "\n", ". ", " "]

# C0/C1 control chars (incl. NUL) except tab/newline — PostgreSQL text rejects NUL.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass
class ChunkDraft:
    """A finalized chunk text paired with its embedding."""

    text: str
    embedding: list[float]


def approx_tokens(text: str) -> int:
    """Approximate token count from character length (~4 chars/token)."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, guarding against zero-norm vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _atomic_splits(text: str, separators: list[str], max_chars: int) -> list[str]:
    """Split ``text`` into pieces each <= ``max_chars`` (best effort).

    Tries the highest-priority separator that occurs in ``text``; pieces still
    too large are recursively split on the next separator, then hard-split on
    characters. Empty pieces are dropped.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Pick the first separator present in the text.
    sep = None
    rest: list[str] = []
    for i, s in enumerate(separators):
        if s and s in text:
            sep = s
            rest = separators[i + 1 :]
            break

    if sep is None:
        # No separator left: hard char split.
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    pieces: list[str] = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            pieces.append(part)
        else:
            pieces.extend(_atomic_splits(part, rest, max_chars))
    return pieces


def _merge_splits(splits: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack atomic ``splits`` into chunks, with a bounded overlap tail.

    Sliding-window merge: accumulate pieces until the next would exceed
    ``max_chars``, emit the chunk, then pop pieces off the front until the
    retained tail is <= ``overlap_chars`` — that tail becomes the overlap prefix
    of the next chunk. A chunk may exceed ``max_chars`` by at most the overlap
    tail plus one piece (standard recursive-splitter behaviour); all atomic
    pieces are already <= ``max_chars``.
    """
    sep = " "  # atomic pieces are re-joined with a single space
    sep_len = len(sep)
    chunks: list[str] = []
    current: list[str] = []
    total = 0  # length of current incl. separators

    for d in splits:
        addition = len(d) + (sep_len if current else 0)
        if current and total + addition > max_chars:
            chunks.append(sep.join(current))
            # Pop from the front until the retained tail fits the overlap budget
            # (or until a single oversized piece is all that remains).
            while current and (total > overlap_chars or total + addition > max_chars):
                removed = len(current[0]) + (sep_len if len(current) > 1 else 0)
                total -= removed
                current = current[1:]
        current.append(d)
        total += len(d) + (sep_len if len(current) > 1 else 0)

    if current:
        chunks.append(sep.join(current))
    return chunks


def chunk_text(
    text: str, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[ChunkDraft]:
    """Recursively split ``text`` into bounded chunks and embed them.

    Returns ``ChunkDraft``s (text + L2-normalized embedding). Every chunk is at
    most ``chunk_chars`` (~300 tokens), so the batched embedding call can never
    exceed the model context.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Defensive: strip NUL/control bytes (PostgreSQL text rejects NUL). The PDF
    # path already does this in clean_pdf_text.
    text = _CONTROL_CHARS.sub("", text)

    atoms = _atomic_splits(text, SEPARATORS, chunk_chars)
    chunk_strs = _merge_splits(atoms, chunk_chars, overlap_chars)
    if not chunk_strs:
        return []

    LOGGER.info(
        f"Recursive split produced {len(chunk_strs)} chunks "
        f"(~{CHUNK_TOKENS}tok target, ~{OVERLAP_TOKENS}tok overlap)."
    )

    vectors = _l2_normalize(np.array(embedder(chunk_strs)))
    return [
        ChunkDraft(text=t, embedding=v.tolist()) for t, v in zip(chunk_strs, vectors)
    ]


def extract_keyphrases(text: str) -> list[str]:
    if not text:
        LOGGER.warning("Empty text passed for keyphrases extraction.")
        return []
    try:
        import yake

        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedup_lim=0.1,
            top=10,
        )
    except Exception:
        LOGGER.error("Dependency not installed. 'uv add -r requirements'")
        raise ConfigurationError("Dependency not installed. 'uv add -r requirements'")

    keywords_score = kw_extractor.extract_keywords(text)
    return [kw for kw, score in keywords_score]


def chunk_enrich(
    chunks: list[ChunkDraft],
    source: Literal["PDF", "Reel", "Youtube"],
    id: str,
    doc_id: str | None = None,
) -> list[Chunk]:
    """Promote drafts to stored Chunks, reusing each draft's embedding."""
    finalized_chunks = []
    for idx, draft in enumerate(chunks):
        enriched = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=draft.text,
            embeddings=draft.embedding,
            source_type=source,
            source_id=id,
            chunk_index=idx,
            keyword=extract_keyphrases(draft.text),
            doc_id=doc_id,
        )
        finalized_chunks.append(enriched)
    return finalized_chunks
