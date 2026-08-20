"""Execute a single evaluation query under a given retrieval configuration.

Maps the four compared configurations onto the system's building blocks:

    direct         -> no retrieval (closed-book generation)
    semantic       -> retrieval.search.search(use_vector=True)
    semantic_bm25  -> retrieval.search.search(use_vector=True, use_bm25=True)
    all_three      -> retrieval.search.search(use_vector, use_bm25, use_graph)

Each retrieval configuration runs with ``rerank`` True or False.

The eval harness is **staged and decoupled** (see README.md):
``retrieve_query`` persists the FINAL reranked chunks — post-rerank and
post-graph-expansion, "hot and ready" — plus graph facts and a raw timing.
``generate_answers`` then does *zero* retrieval work: it assembles context
strings locally from the persisted chunks and bundles many (question, context)
pairs per API call. Per-query latency is therefore meaningless in this pipeline
(the report marks it n/a; raw timings remain in the JSON caches for reference).

``src`` imports are function-local so the module imports without a database or
models present.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace


@dataclass
class RetrievalOutcome:
    """Final retrieval state, persisted per (query, config) and reused verbatim."""

    titles: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # chunk texts, final order
    graph_facts: str = ""
    retrieval_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "titles": self.titles,
            "contexts": self.contexts,
            "graph_facts": self.graph_facts,
            "retrieval_ms": self.retrieval_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalOutcome":
        return cls(
            titles=list(data.get("titles", [])),
            contexts=list(data.get("contexts", [])),
            graph_facts=data.get("graph_facts", ""),
            retrieval_ms=float(data.get("retrieval_ms", 0.0)),
        )


def retrieve_query(
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
    query_entities: list[str] | None = None,
) -> RetrievalOutcome:
    """Run retrieval for one question under one configuration.

    Returns the final reranked chunk list (post structural expansion when the
    graph is enabled) + graph facts — exactly what generation needs, so the
    generate phase never touches the retrieval stack. The ``direct`` mode
    performs no retrieval and returns an empty outcome.
    """
    if mode_name == "direct":
        return RetrievalOutcome()

    from retrieval.search import search

    t0 = time.perf_counter()
    result = search(
        question=record["question"],
        use_vector=use_vector,
        use_bm25=use_bm25,
        use_graph=use_graph,
        rerank=rerank,
        top_k=top_k,
        candidate_k=candidate_k,
        query_embedding=query_embedding,
        query_entities=query_entities,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    return RetrievalOutcome(
        titles=[c.source_id for c in result.chunks],
        contexts=[c.text for c in result.chunks],
        graph_facts=result.graph_facts,
        retrieval_ms=retrieval_ms,
    )


def _assemble_context(outcome: RetrievalOutcome) -> str:
    """Rebuild the numbered, citation-ready context string from persisted chunks."""
    from context.builder import build_context

    chunks = [
        SimpleNamespace(
            chunk_id=f"eval-{i}",
            source_id=title,
            text=text,
        )
        for i, (title, text) in enumerate(zip(outcome.titles, outcome.contexts))
    ]
    context, _citations = build_context(chunks, outcome.graph_facts)
    return context


def generate_answers(
    records: list[dict],
    retrievals: dict,
    *,
    batch_size: int = 40,
) -> dict:
    """Generate answers for every (query, config) cell from persisted retrievals.

    Pure assembly + bundled generation — no retrieval work. Returns a nested
    dict ``{qid: {label: {"answer": str, "generation_ms": float, "context": str}}}``
    where ``label`` is the RunSpec label (e.g. ``all_three+rerank``).
    """
    from llm.generator import generate_batch

    questions = {r["id"]: r["question"] for r in records}
    answers: dict = {}

    for qid, question in questions.items():
        answers.setdefault(qid, {})
        for label, cell in retrievals.get(qid, {}).items():
            if label == "direct":
                answers[qid][label] = {"context": ""}
                continue
            outcome = RetrievalOutcome.from_dict(cell)
            answers[qid][label] = {"context": _assemble_context(outcome)}

    # One bundled generation call per (label, batch of questions) — ~100 pairs
    # per label -> ~3 calls at batch_size=40 (18 RAG + 3 direct calls total).
    labels = {label for qid in answers for label in answers[qid]}
    for label in labels:
        qids = [qid for qid in questions if label in answers[qid]]
        cells = [answers[qid][label] for qid in qids]
        pairs = [(questions[qid], cells[i]["context"]) for i, qid in enumerate(qids)]
        t0 = time.perf_counter()
        generated = generate_batch(pairs, batch_size=batch_size)
        gen_ms = (time.perf_counter() - t0) * 1000.0
        for qid, answer in zip(qids, generated):
            answers[qid][label]["answer"] = answer
            answers[qid][label]["generation_ms"] = gen_ms

    return answers