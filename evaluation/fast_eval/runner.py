"""Concurrent runner for fast_eval: the full ablation with real latency.

Runs the 7-configuration matrix — ``direct`` + (``semantic`` / ``semantic_bm25`` /
``all_three``) × {rerank off, rerank on} — over all queries. Each query is
retrieved **and** generated interactively (not bundled), so per-question
retrieval / generation / total latency is real and measurable.

Concurrency: a single ``ThreadPoolExecutor`` (``MAX_WORKERS``) processes cells. Each
cell's retrieval + generation is independent; the generator and stores open their own
connections per call, so threaded use is safe.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from evaluation import metrics
from evaluation.fast_eval import config
from evaluation.fast_eval.llm import (
    CallTracker,
    extract_query_entities,
    generate_answer,
)

# Minimum seconds between entity-extraction calls (serialized). b.ai throttles
# sustained bursts (429); 1/sec keeps the 100-call phase under the ceiling.
_ENTITY_MIN_INTERVAL = 1.0


def _entities_sequential(
    record: dict, tracker: CallTracker
) -> tuple[str, list[str]]:
    """Extract one query's entities with the shared tracker (serialized loop)."""
    return record["id"], extract_query_entities(record["question"], tracker=tracker)


@dataclass(frozen=True)
class Spec:
    """One row of the ablation matrix."""
    mode: str
    rerank: bool
    use_vector: bool
    use_bm25: bool
    use_graph: bool

    @property
    def label(self) -> str:
        return self.mode if self.mode == "direct" else f"{self.mode}{'+rerank' if self.rerank else ''}"


@dataclass
class CellResult:
    qid: str
    spec: Spec
    answer: str
    f1: float
    em: float
    recall: float
    hit_at_k: float | None
    answer_in_context: float | None
    retrieval_ms: float
    generation_ms: float
    total_ms: float


def build_specs() -> list[Spec]:
    return [
        Spec("direct", False, False, False, False),
        Spec("semantic", False, True, False, False),
        Spec("semantic", True, True, False, False),
        Spec("semantic_bm25", False, True, True, False),
        Spec("semantic_bm25", True, True, True, False),
        Spec("all_three", False, True, True, True),
        Spec("all_three", True, True, True, True),
    ]


def _run_cell(
    record: dict,
    spec: Spec,
    *,
    query_embedding: list[float],
    query_entities: list[str],
    top_k: int,
    candidate_k: int,
    tracker: CallTracker,
) -> CellResult:
    """Run retrieval + generation for one (query, spec) cell, timing each stage."""
    from retrieval.search import search

    question = record["question"]
    gold = record["answer"]
    gold_titles = record["supporting_titles"]

    if spec.mode == "direct":
        answer, gen_ms = generate_answer(question, closed_book=True, tracker=tracker)
        f1 = metrics.f1_score(answer, gold)
        em = metrics.exact_match(answer, gold)
        recall = metrics.token_recall(answer, gold)
        return CellResult(
            record["id"], spec, answer, f1, em, recall, None, None, 0.0, gen_ms, gen_ms,
        )

    t0 = time.perf_counter()
    result = search(
        question,
        use_vector=spec.use_vector,
        use_bm25=spec.use_bm25,
        use_graph=spec.use_graph,
        rerank=spec.rerank,
        top_k=top_k,
        candidate_k=candidate_k,
        query_embedding=query_embedding if spec.use_vector else None,
        query_entities=query_entities if spec.use_graph else None,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000.0
    if spec.rerank:
        tracker.record("rerank", 0.0)  # one Jina rerank call per rerank cell

    contexts = [c.text for c in result.chunks]
    if result.graph_facts:
        contexts.append(result.graph_facts)

    from context.builder import build_context
    context_str, _citations = build_context(result.chunks, result.graph_facts)

    answer, gen_ms = generate_answer(question, context_str, tracker=tracker)

    f1 = metrics.f1_score(answer, gold)
    em = metrics.exact_match(answer, gold)
    recall = metrics.token_recall(answer, gold)
    hit = metrics.retrieval_hit_at_k([c.source_id for c in result.chunks], gold_titles)
    aic = metrics.answer_in_context(gold, contexts)

    return CellResult(
        record["id"], spec, answer, f1, em, recall, hit, aic,
        retrieval_ms, gen_ms, retrieval_ms + gen_ms,
    )


def run(
    records: list[dict],
    *,
    top_k: int = config.TOP_K,
    candidate_k: int = config.CANDIDATE_K,
    max_workers: int = config.MAX_WORKERS,
    on_cell=None,
    on_spec_start=None,
) -> tuple[list[CellResult], CallTracker]:
    """Run the full ablation concurrently. Returns (cells, tracker)."""
    from embeddings.embedder import embedder

    tracker = CallTracker()
    specs = build_specs()

    # Query embeddings: ONE batched call for all questions (Mistral), not per query.
    questions = [r["question"] for r in records]
    ids = [r["id"] for r in records]
    embeddings = embedder(questions)
    tracker.record("embedding", 0.0)  # one batched embedding call (not an LLM)
    emb_by_id = {qid: emb for qid, emb in zip(ids, embeddings)}

    # Query entities: extract once per query (DeepSeek v4-flash), reused by both
    # all_three cells. One entity LLM call per query. Serialized AND spaced to
    # stay under the provider's burst ceiling: 100 calls ~1/sec each rides the
    # throttle instead of tripping it (b.ai 429s under sustained bursts).
    ent_by_id: dict[str, list[str]] = {}
    _last_ent_call = 0.0
    for record in records:
        wait = _ENTITY_MIN_INTERVAL - (time.monotonic() - _last_ent_call)
        if wait > 0:
            time.sleep(wait)
        qid, ents = _entities_sequential(record, tracker)
        _last_ent_call = time.monotonic()
        ent_by_id[qid] = ents

    cells: list[CellResult] = []
    for spec in specs:
        if on_spec_start:
            on_spec_start(spec.label, len(records))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _run_cell,
                    record,
                    spec,
                    query_embedding=emb_by_id[record["id"]],
                    query_entities=ent_by_id.get(record["id"], []),
                    top_k=top_k,
                    candidate_k=candidate_k,
                    tracker=tracker,
                ): record
                for record in records
            }
            done = 0
            for fut in as_completed(futures):
                cell = fut.result()
                cells.append(cell)
                done += 1
                if on_cell:
                    on_cell(spec.label, done, len(records))

    return cells, tracker
