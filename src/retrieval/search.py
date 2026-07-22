"""Composable, explicitly-forced retrieval (vector / BM25 / graph).

Unlike ``reasoning.router`` (heuristic, and slated to become agentic later),
this module performs **no** automatic strategy selection: the caller states
exactly which signals to combine. It is the retrieval entry point used by the
evaluation harness to benchmark each retrieval configuration in isolation.

Uses the new document-level clustering pipeline:
  Stage 2: doc_level_soft_rank (summary ANN + question ANN + entity boost)
  Stage 3: chunk-level search (dense + optionally lexical) restricted to top docs
  Stage 4: cross-list RRF fusion (doc-level + dense + lexical)
  Stage 5: rerank
  Stage 6: structural expansion (small-to-big graph traversal)
  Stage 7: graph facts (multi-hop BFS entity context)

Configurations used by the evaluation:
    * semantic only   -> ``use_vector=True``
    * semantic + bm25 -> ``use_vector=True, use_bm25=True`` (fused with RRF)
    * all three       -> ``use_vector=True, use_bm25=True, use_graph=True``
    * direct baseline -> none of them (no retrieval; handled by the caller)

Each configuration runs with ``rerank`` on or off.
"""

from dataclasses import dataclass, field

from config.settings import get_settings
from constants.logger import setup_logger
from embeddings.embedder import embedder
from retrieval.pgvector import (
    RetrievedChunk,
    doc_level_soft_rank,
    _chunk_dense_search,
    _chunk_lexical_search,
    _rrf_score,
)
from retrieval.kuzu_store import get_entity_context, structural_expansion
from retrieval.reranker import rerank as _rerank

LOGGER = setup_logger(__name__)
settings = get_settings()


@dataclass
class RetrievalResult:
    """Outcome of a forced retrieval: ranked chunks + optional graph facts."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    graph_facts: str = ""


def extract_query_entities(question: str) -> list[str]:
    """Hybrid query entity extraction for graph seeding: YAKE first, LLM fallback.

    Thin delegator to the canonical implementation in ``graph.entity_extraction``.
    """
    from graph.entity_extraction import extract_query_entities as _impl

    return _impl(question)


def _rrf_fuse(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Reciprocal-Rank-Fusion of multiple ranked chunk lists."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + _rrf_score(
                rank, k
            )
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_id, score in ranked:
        c = chunk_map[chunk_id]
        c.score = score
        fused.append(c)
    return fused


def search(
    question: str,
    *,
    use_vector: bool = True,
    use_bm25: bool = False,
    use_graph: bool = False,
    rerank: bool = False,
    top_k: int = 5,
    candidate_k: int = 20,
    query_embedding: list[float] | None = None,
) -> RetrievalResult:
    """Run an explicitly-configured retrieval and return the top-``top_k`` chunks.

    Uses the document-level clustering pipeline:
      doc-level soft rank -> chunk search (dense + optional lexical)
      -> cross-list RRF fusion -> rerank -> structural expansion -> graph facts

    Falls back to flat (non-doc-routed) search when no document clusters exist,
    preserving backward compatibility with corpora ingested before the clustering
    pipeline was introduced.
    """
    emb = query_embedding if query_embedding is not None else embedder([question])[0]

    # Stage 2: Doc-level soft rank (summary ANN + question ANN + entity boost).
    doc_ranking = doc_level_soft_rank(emb, question, top_k=25)
    doc_ids = [d[0] for d in doc_ranking]
    doc_rank_map = (
        {d[0]: i + 1 for i, d in enumerate(doc_ranking)} if doc_ranking else None
    )

    # Stage 3 + 4: Chunk-level search + cross-list RRF fusion.
    # For each chunk, the fused score combines:
    #   A = doc-level rank signal (from Stage 2)
    #   B = dense chunk rank (cosine ANN)
    #   C = lexical chunk rank (tsvector BM25, if use_bm25)
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    if use_vector:
        dense = _chunk_dense_search(emb, doc_ids, top_k=candidate_k)
        for rank, c in enumerate(dense, start=1):
            a = _rrf_score(doc_rank_map.get(c.doc_id, 1)) if doc_rank_map else 0
            scores[c.chunk_id] = a + _rrf_score(rank)  # A + B
            chunk_map[c.chunk_id] = c

    if use_bm25:
        lexical = _chunk_lexical_search(question, doc_ids, top_k=candidate_k)
        for rank, c in enumerate(lexical, start=1):
            c_score = _rrf_score(rank)  # C
            if c.chunk_id in scores:
                scores[c.chunk_id] += c_score  # was A+B, now A+B+C
            else:
                a = _rrf_score(doc_rank_map.get(c.doc_id, 1)) if doc_rank_map else 0
                scores[c.chunk_id] = a + c_score  # A + C
                chunk_map[c.chunk_id] = c

    # Fallback: flat search when no document clusters exist.
    if not scores and (use_vector or use_bm25):
        LOGGER.info("No doc-level clusters; falling back to flat chunk search.")
        result_lists: list[list[RetrievedChunk]] = []
        if use_vector:
            dense = _chunk_dense_search(emb, None, top_k=candidate_k)
            result_lists.append(dense)
        if use_bm25:
            lexical = _chunk_lexical_search(question, None, top_k=candidate_k)
            result_lists.append(lexical)
        fused = _rrf_fuse(result_lists) if result_lists else []
        for rank, c in enumerate(fused[:candidate_k], start=1):
            scores[c.chunk_id] = float(c.score)
            chunk_map[c.chunk_id] = c

    if not scores:
        LOGGER.info("No chunks retrieved.")
        return RetrievalResult(chunks=[], graph_facts="")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
    chunks: list[RetrievedChunk] = []
    for chunk_id, score in ranked:
        c = chunk_map[chunk_id]
        c.score = score
        chunks.append(c)

    # Stage 5: cross-encoder rerank.
    if rerank and chunks:
        chunks = _rerank(question, chunks, top_k=top_k)
    else:
        chunks = chunks[:top_k]

    # Stage 6 + 7: structural expansion + graph facts.
    graph_facts = ""
    if use_graph and chunks:
        seed_ids = [c.chunk_id for c in chunks]
        siblings = structural_expansion(seed_ids)
        if siblings:
            existing = {c.chunk_id for c in chunks}
            for chunk in siblings:
                if chunk.chunk_id not in existing:
                    chunks.append(chunk)
                    existing.add(chunk.chunk_id)

        entities = extract_query_entities(question)
        if entities:
            graph_facts = get_entity_context(entities)

    LOGGER.info(
        f"Forced search [vector={use_vector} bm25={use_bm25} graph={use_graph} "
        f"rerank={rerank}] -> {len(chunks)} chunks, "
        f"graph_facts={'yes' if graph_facts else 'no'}."
    )
    return RetrievalResult(chunks=chunks, graph_facts=graph_facts)
