"""Semantic chunking with pooled embeddings and a re-embedding pass for splits.

Paragraph embeddings are computed once in :func:`chunk_text` and feed both
clustering and the pooled vectors for whole-cluster chunks (length-weighted mean
pooling + L2 renormalization). Oversized clusters are subdivided into
sentence-level chunks whose pooled vectors are only a loose approximation of the
re-sliced text, so those chunks get a second, direct embedding pass before
storage. Chunks are stored and searched under cosine distance
(``vector_cosine_ops``), so only direction matters and every stored vector is
L2-normalized to stay on the same unit sphere as freshly embedded queries.
"""

from dataclasses import dataclass
from typing import Literal
import uuid

import numpy as np

from constants.logger import setup_logger
from ingestion.normalize import split_into_paragraphs
from constants.exceptions import ModelError, ConfigurationError
from embeddings.embedder import embedder
from ingestion.chunk_schema import Chunk

LOGGER = setup_logger(__name__)


@dataclass
class ChunkDraft:
    """A finalized chunk text paired with its reused (non-recomputed) embedding."""

    text: str
    embedding: list[float]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, guarding against zero-norm vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _pool(vectors: np.ndarray, weights: np.ndarray | None = None) -> list[float]:
    """Length-weighted mean pool of paragraph vectors, then L2 renormalize.

    Approximates the embedding of the concatenated text by reusing the
    component paragraph embeddings instead of calling the model again. Cosine
    search ignores magnitude, so renormalizing keeps the result on the same
    unit sphere as query embeddings.
    """
    if len(vectors) == 1:
        pooled = vectors[0].astype(float)
    elif weights is None:
        pooled = vectors.mean(axis=0)
    else:
        w = weights.astype(float)
        total = w.sum()
        w = w / total if total > 0 else np.full(len(w), 1.0 / len(w))
        pooled = (vectors * w[:, None]).sum(axis=0)

    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.tolist()


def clustering(normalized: np.ndarray, threshold: float = 0.7) -> list[list[int]]:
    """Group consecutive paragraphs into topic segments by cosine similarity.
    Expects L2-normalized embeddings (the caller normalizes once upstream), so
    the
    dot product below is exactly cosine similarity. Returns clusters as lists
    of
    paragraph indices, letting the caller reuse the paragraph texts and
    vectors.
    """

    n = len(normalized)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

    clusters: list[list[int]] = []
    current = [0]
    for idx, sim in enumerate(similarities):
        if sim > threshold:
            current.append(idx + 1)
        else:
            clusters.append(current)
            current = [idx + 1]
    clusters.append(current)

    LOGGER.info(f"Created {len(clusters)} clusters from {n} paragraphs.")
    return clusters


def _emit_subchunk(
    sentences: list[str],
    sent_para: list[int],
    embeddings: np.ndarray,
    start: int,
    end: int,
) -> ChunkDraft:
    """Build a sub-chunk from ``sentences[start:end]``, pooling the embeddings of
    the paragraphs those sentences came from (weighted by words contributed)."""
    text = " ".join(sentences[start:end])
    covered = sorted({sent_para[i] for i in range(start, end)})
    local = {p: k for k, p in enumerate(covered)}
    weights = np.zeros(len(covered), dtype=float)
    for i in range(start, end):
        weights[local[sent_para[i]]] += len(sentences[i].split())
    return ChunkDraft(text=text, embedding=_pool(embeddings[covered], weights))


def split_large_chunks(
    paragraphs: list[str],
    embeddings: np.ndarray,
    chunk_length: int = 300,
    overlap: int = 2,
) -> list[ChunkDraft]:
    """Subdivide one oversized cluster into chunks.

    Cut points prefer the weakest *paragraph* similarity boundary inside a short
    look-ahead window (reusing the clustering scores), and fall back to the
    entity-discontinuity heuristic when no paragraph boundary is in range. Each
    emitted chunk's embedding is a pooled approximation of the paragraph vectors
    it covers; :func:`chunk_text` re-embeds these chunk texts directly in a
    second pass before storage.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        LOGGER.error("Unable to fetch model: en_core_web_sm")
        raise ModelError(f"Unable to fetch model: {e}")

    # Sentences + per-sentence entities, remembering the source paragraph so
    # each sub-chunk can map back to the correct embeddings.
    sentences: list[str] = []
    sent_entities: list[set[str]] = []
    sent_para: list[int] = []
    for p_idx, doc in enumerate(nlp.pipe(paragraphs)):
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            sentences.append(s)
            sent_entities.append({ent.text.lower() for ent in sent.ents})
            sent_para.append(p_idx)

    if not sentences:
        return []

    # Consecutive-paragraph cosine similarities. Every value is above the
    # clustering threshold by definition, but the relatively weakest links are
    # the best places to cut an oversized cluster.
    normalized = _l2_normalize(embeddings)
    if len(normalized) > 1:
        para_sim = np.sum(normalized[:-1] * normalized[1:], axis=1)
    else:
        para_sim = np.array([])

    drafts: list[ChunkDraft] = []
    start = 0
    while start < len(sentences):
        word_count = 0
        end = start
        while end < len(sentences) and word_count < chunk_length:
            word_count += len(sentences[end].split())
            end += 1

        if end >= len(sentences):
            drafts.append(_emit_subchunk(sentences, sent_para, embeddings, start, len(sentences)))
            break

        window = range(end, min(end + 5, len(sentences)))
        cut = end
        effective_overlap = overlap

        # Primary signal: cut at the weakest paragraph boundary in the window.
        weakest_sim = None
        weakest_cut = None
        for i in window:
            if sent_para[i] != sent_para[i - 1]:
                b = sent_para[i - 1]
                sim = para_sim[b] if 0 <= b < len(para_sim) else 1.0
                if weakest_sim is None or sim < weakest_sim:
                    weakest_sim = sim
                    weakest_cut = i

        if weakest_cut is not None:
            cut = weakest_cut
        else:
            # Fallback: original entity-continuity heuristic.
            window_entities: set[str] = set()
            for i in range(start, end):
                window_entities |= sent_entities[i]
            if not window_entities:
                effective_overlap = overlap + 2
            else:
                for i in window:
                    if not sent_entities[i] & window_entities:
                        cut = i
                        break

        drafts.append(_emit_subchunk(sentences, sent_para, embeddings, start, cut))
        start = max(cut - effective_overlap, start + 1)

    LOGGER.info(f"Split oversized cluster into {len(drafts)} chunks.")
    return drafts


def chunk_text(text: str, threshold: float = 0.7, chunk_length: int = 300) -> list[ChunkDraft]:
    """Cluster paragraphs semantically and return finalized chunk drafts.

    Embeds paragraphs once; those vectors drive clustering and the pooled
    embeddings of whole-cluster chunks. Oversized clusters are split into
    sentence-level chunks (still pooled internally), and because that slicing
    crosses paragraph boundaries, the split chunks are re-embedded directly in a
    single batched second pass so their stored vectors reflect their actual text.
    """
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    LOGGER.info("Embedding paragraphs once (reused for clustering + pooled chunk vectors).")
    embeddings = np.array(embedder(paragraphs))
    normalized = _l2_normalize(embeddings)

    clusters = clustering(normalized, threshold)

    drafts: list[ChunkDraft] = []
    split_positions: list[int] = []  # indices into `drafts` produced by splitting
    for cluster in clusters:
        cluster_embeddings = embeddings[cluster]
        cluster_paragraphs = [paragraphs[i] for i in cluster]
        word_count = sum(len(p.split()) for p in cluster_paragraphs)

        if word_count > chunk_length:
            split_drafts = split_large_chunks(cluster_paragraphs, cluster_embeddings, chunk_length)
            split_positions.extend(range(len(drafts), len(drafts) + len(split_drafts)))
            drafts.extend(split_drafts)
        else:
            weights = np.array([len(p.split()) for p in cluster_paragraphs], dtype=float)
            drafts.append(
                ChunkDraft(
                    text=" ".join(cluster_paragraphs),
                    embedding=_pool(cluster_embeddings, weights),
                )
            )

    # Second embedding round: split chunks slice across paragraph/sentence
    # boundaries, so their pooled vectors are the loosest approximation. Re-embed
    # those chunk texts directly (one batched call) and replace the pooled
    # placeholder; whole-cluster chunks keep their pooled embedding. Renormalize
    # so every stored vector stays unit-length under cosine search.
    if split_positions:
        LOGGER.info(f"Re-embedding {len(split_positions)} split chunks (second pass).")
        reembedded = _l2_normalize(np.array(embedder([drafts[i].text for i in split_positions])))
        for pos, vector in zip(split_positions, reembedded):
            drafts[pos].embedding = vector.tolist()

    LOGGER.info(f"Final chunk count: {len(drafts)}")
    return drafts


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


def chunk_enrich(chunks: list[ChunkDraft], source: Literal["PDF", "Reel", "Youtube"], id: str) -> list[Chunk]:
    """Promote drafts to stored Chunks, reusing each draft's pooled embedding."""
    finalized_chunks = []
    for idx, draft in enumerate(chunks):
        enriched = Chunk(
            chunk_id=str(uuid.uuid1()),
            text=draft.text,
            embeddings=draft.embedding,
            source_type=source,
            source_id=id,
            chunk_index=idx,
            keyword=extract_keyphrases(draft.text),
        )
        finalized_chunks.append(enriched)
    return finalized_chunks
