"""Orchestrate the full evaluation: every configuration over every query.

Run matrix (7 configurations):
    direct                                  (rerank N/A)
    semantic        x {rerank, no-rerank}
    semantic_bm25   x {rerank, no-rerank}
    all_three       x {rerank, no-rerank}

For each configuration the harness scores F1, EM, answer recall, retrieval
hit@k ("top"), answer-in-context, and latency (retrieval / generation / total).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from __future__ import annotations

from evaluation import config, metrics
from evaluation.modes import run_query


@dataclass(frozen=True)
class RunSpec:
    """One row of the run matrix."""

    mode_name: str
    rerank: bool
    use_vector: bool
    use_bm25: bool
    use_graph: bool

    @property
    def label(self) -> str:
        if self.mode_name == config.DIRECT_MODE:
            return "direct"
        return f"{self.mode_name}{'+rerank' if self.rerank else ''}"


@dataclass
class RunResult:
    mode: str
    rerank: bool
    n: int
    top_k: int
    f1: float
    em: float
    recall: float
    hit_at_k: float | None
    answer_in_context: float | None
    graph_lift: float | None = None
    latency: dict = field(default_factory=dict)
    retrieval_latency: dict = field(default_factory=dict)
    generation_latency: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_run_specs(modes: list[config.ModeSpec] | None = None) -> list[RunSpec]:
    """The 7-row run matrix: direct + (3 retrieval modes x rerank on/off)."""
    modes = modes if modes is not None else config.RETRIEVAL_MODES
    specs: list[RunSpec] = [
        RunSpec(
            config.DIRECT_MODE,
            rerank=False,
            use_vector=False,
            use_bm25=False,
            use_graph=False,
        )
    ]
    for mode in modes:
        for rerank in (False, True):
            specs.append(
                RunSpec(
                    mode.name,
                    rerank=rerank,
                    use_vector=mode.use_vector,
                    use_bm25=mode.use_bm25,
                    use_graph=mode.use_graph,
                )
            )
    return specs


def _score_run(
    spec: RunSpec, outcomes: list[tuple[dict, object]], top_k: int
) -> RunResult:
    f1s, ems, recalls = [], [], []
    hits, ans_in_ctx = [], []
    retrieval_ms, generation_ms, total_ms = [], [], []

    is_retrieval = spec.mode_name != config.DIRECT_MODE
    for record, outcome in outcomes:
        gold = record["answer"]
        f1s.append(metrics.f1_score(outcome.answer, gold))
        ems.append(metrics.exact_match(outcome.answer, gold))
        recalls.append(metrics.token_recall(outcome.answer, gold))
        if is_retrieval:
            hits.append(
                metrics.retrieval_hit_at_k(
                    outcome.retrieved_titles, record["supporting_titles"]
                )
            )
            ans_in_ctx.append(metrics.answer_in_context(gold, outcome.contexts))
        retrieval_ms.append(outcome.retrieval_ms)
        generation_ms.append(outcome.generation_ms)
        total_ms.append(outcome.retrieval_ms + outcome.generation_ms)

    return RunResult(
        mode=spec.mode_name,
        rerank=spec.rerank,
        n=len(outcomes),
        top_k=top_k,
        f1=metrics.mean(f1s),
        em=metrics.mean(ems),
        recall=metrics.mean(recalls),
        hit_at_k=metrics.mean(hits) if hits else None,
        answer_in_context=metrics.mean(ans_in_ctx) if ans_in_ctx else None,
        latency=metrics.aggregate_latency(total_ms),
        retrieval_latency=metrics.aggregate_latency(retrieval_ms),
        generation_latency=metrics.aggregate_latency(generation_ms),
    )


def run_all(
    records: list[dict],
    *,
    top_k: int = config.TOP_K,
    candidate_k: int = config.CANDIDATE_K,
    modes: list[config.ModeSpec] | None = None,
    on_progress=None,
) -> list[RunResult]:
    """Run the whole matrix and return one :class:`RunResult` per configuration.

    Question embeddings are computed once (a single batched pass) and reused
    across every vector-using configuration to keep the comparison fair and
    fast. ``on_progress(spec_label, done, total)`` is called per query if given.
    """
    from embeddings.embedder import embedder

    questions = [r["question"] for r in records]
    embeddings = embedder(questions)  # one batched embedding pass
    id_to_embedding = {r["id"]: emb for r, emb in zip(records, embeddings)}

    specs = build_run_specs(modes)
    # comment above/below to use all or only three.
    """specs = [
        RunSpec(
            mode_name="all_three",
            rerank=False,
            use_vector=True,
            use_bm25=True,
            use_graph=True,
        ),
        RunSpec(
            mode_name="all_three",
            rerank=True,
            use_vector=True,
            use_bm25=True,
            use_graph=True,
        ),
    ]"""
    results: list[RunResult] = []

    for spec in specs:
        outcomes: list[tuple[dict, object]] = []
        for i, record in enumerate(records, start=1):
            query_embedding = id_to_embedding[record["id"]] if spec.use_vector else None
            outcome = run_query(
                record,
                spec.mode_name,
                use_vector=spec.use_vector,
                use_bm25=spec.use_bm25,
                use_graph=spec.use_graph,
                rerank=spec.rerank,
                top_k=top_k,
                candidate_k=candidate_k,
                query_embedding=query_embedding,
            )
            outcomes.append((record, outcome))
            if on_progress:
                on_progress(spec.label, i, len(records))
        results.append(_score_run(spec, outcomes, top_k))

    # Compute graph lift: for each retrieval mode, compare F1 with/without graph.
    # Semantic_bm25 and semantic share the same no-graph baseline; all_three is the
    # only mode that adds graph. Compare all_three vs semantic_bm25 (same BM25+vector
    # foundation, one adds graph). Delta per query then averaged.
    all_three_results = [r for r in results if r.mode == "all_three"]
    semantic_bm25_results = [r for r in results if r.mode == "semantic_bm25"]
    for graph_run in all_three_results:
        paired = [r for r in semantic_bm25_results if r.rerank == graph_run.rerank]
        if paired:
            graph_run.graph_lift = graph_run.f1 - paired[0].f1

    return results
