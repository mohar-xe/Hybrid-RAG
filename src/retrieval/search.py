"""Composable, explicitly-forced retrieval (vector / BM25 / graph).

Unlike ``reasoning.router`` (heuristic, and slated to become agentic later),
this module performs **no** automatic strategy selection: the caller states
exactly which signals to combine. It is the retrieval entry point used by the
evaluation harness to benchmark each retrieval configuration in isolation, and
it makes the BM25 path reachable again without resurrecting the deferred router.

Configurations used by the evaluation:
    * semantic only   -> ``use_vector=True``
    * semantic + bm25 -> ``use_vector=True, use_bm25=True``  (fused with RRF)
    * all three       -> ``use_vector=True, use_bm25=True, use_graph=True``
    * direct baseline -> none of them (no retrieval; handled by the caller)

Each configuration runs with ``rerank`` on or off.
"""

from dataclasses import dataclass, field

from config.settings import get_settings
from constants.logger import setup_logger
from embeddings.embedder import embedder
from retrieval.pgvector import RetrievedChunk, vector_search, bm25_search, _rrf_score
from retrieval.kuzu_store import get_entity_context
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

    Thin delegator to the canonical implementation in ``graph.entity_extraction``
    (YAKE keyphrases, with a DeepSeek LLM fallback when YAKE is empty). Kept here
    so existing importers — ``from retrieval.search import extract_query_entities``,
    used by the evaluation harness — keep working unchanged.
    """
    from graph.entity_extraction import extract_query_entities as _impl

    return _impl(question)


def _rrf_fuse(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal-Rank-Fusion of multiple ranked chunk lists.

    Each chunk's fused score is the sum of ``1/(k+rank)`` over the lists it
    appears in; chunks are returned highest-fused-score first with ``.score``
    overwritten by the fused value.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + _rrf_score(rank, k)
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

    Args:
        question: the user query.
        use_vector: include dense (pgvector cosine) retrieval.
        use_bm25: include sparse (Postgres full-text) retrieval.
        use_graph: additionally fetch Kùzu graph facts for the query entities.
        rerank: cross-encoder rerank the fused candidates down to ``top_k``;
            when False the top-``top_k`` by retrieval score are kept as-is.
        top_k: number of chunks to return.
        candidate_k: per-retriever candidate pool size before fusion/rerank.
        query_embedding: precomputed query embedding (skips re-embedding when
            the caller already has one).

    Returns:
        ``RetrievalResult`` with the ranked chunks and any graph facts.
    """
    result_lists: list[list[RetrievedChunk]] = []

    if use_vector:
        emb = query_embedding if query_embedding is not None else embedder([question])[0]
        result_lists.append(vector_search(emb, top_k=candidate_k))
    if use_bm25:
        result_lists.append(bm25_search(question, top_k=candidate_k))

    if len(result_lists) > 1:
        chunks = _rrf_fuse(result_lists)
    elif result_lists:
        chunks = result_lists[0]
    else:
        chunks = []

    if rerank and chunks:
        chunks = _rerank(question, chunks, top_k=top_k)
    else:
        chunks = chunks[:top_k]

    graph_facts = ""
    if use_graph:
        entities = extract_query_entities(question)
        if entities:
            graph_facts = get_entity_context(entities)

    LOGGER.info(
        f"Forced search [vector={use_vector} bm25={use_bm25} graph={use_graph} "
        f"rerank={rerank}] -> {len(chunks)} chunks, graph_facts={'yes' if graph_facts else 'no'}."
    )
    return RetrievalResult(chunks=chunks, graph_facts=graph_facts)
