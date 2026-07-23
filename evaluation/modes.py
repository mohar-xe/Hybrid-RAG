"""Execute a single evaluation query under a given retrieval configuration.

Maps the four compared configurations onto the system's building blocks:

    direct         -> llm.generator.generate(..., closed_book=True)   (no retrieval)
    semantic       -> retrieval.search.search(use_vector=True)
    semantic_bm25  -> retrieval.search.search(use_vector=True, use_bm25=True)
    all_three      -> retrieval.search.search(use_vector, use_bm25, use_graph)

Each retrieval configuration runs with ``rerank`` True or False. ``src`` imports
are function-local so the module imports without a database/models present.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class QueryOutcome:
    answer: str
    retrieved_titles: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0


def run_query(
    record: dict,
    mode_name: str,
    *,
    use_vector: bool,
    use_bm25: bool,
    use_graph: bool,
    rerank: bool,
    top_k: int,
    candidate_k: int,
    query_embedding: list[float] | None = None,
) -> QueryOutcome:
    """Run one question end-to-end and capture answer, retrieved titles, latency."""
    from llm.generator import generate

    question = record["question"]

    # Closed-book baseline: parametric knowledge only, no retrieval.
    if mode_name == "direct":
        t0 = time.perf_counter()
        answer = generate(question, closed_book=True, eval_mode=True)
        gen_ms = (time.perf_counter() - t0) * 1000.0
        return QueryOutcome(answer=str(answer), generation_ms=gen_ms)

    from retrieval.search import search
    from context.builder import build_context

    t0 = time.perf_counter()
    result = search(
        question,
        use_vector=use_vector,
        use_bm25=use_bm25,
        use_graph=use_graph,
        rerank=rerank,
        top_k=top_k,
        candidate_k=candidate_k,
        query_embedding=query_embedding,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    context, _citations = build_context(result.chunks, result.graph_facts)

    t1 = time.perf_counter()
    answer = generate(question, context, eval_mode=True)
    generation_ms = (time.perf_counter() - t1) * 1000.0

    return QueryOutcome(
        answer=str(answer),
        retrieved_titles=[c.source_id for c in result.chunks],
        contexts=[c.text for c in result.chunks]
        + ([result.graph_facts] if result.graph_facts else []),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )
