"""Orchestrate the staged evaluation: artifacts -> retrieve -> generate -> report.

Run matrix (7 configurations):
    direct                                  (rerank N/A)
    semantic        x {rerank, no-rerank}
    semantic_bm25   x {rerank, no-rerank}
    all_three       x {rerank, no-rerank}

The harness is **decoupled into phases**, each persisted to JSON under
``evaluation/data/eval_cache/`` so the run is resumable and each phase can be
recomputed independently (``--force``):

    artifacts  — query embeddings (ONE batched Mistral call) + query entities
                 (ONE bundled Gemini call) for all questions, cached.
    retrieve   — every (config x query) cell: final reranked chunks + graph
                 facts persisted "hot and ready"; no LLM generation involved
                 (only the Jina reranker on the 6 RAG configs).
    generate   — bundles ~40 (question, context) pairs per Gemini call, reading
                 ONLY the persisted retrievals (zero retrieval work).
    report     — scores answers + retrieval from the caches, writes markdown.

Per-query latency is **not a metric** in this pipeline (decoupled phases +
bundled generation) — the report renders it n/a and explains why; raw timings
remain in the JSON caches for reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from evaluation import config, metrics, persist
from evaluation.modes import generate_answers, retrieve_query

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def artifacts(
    records: list[dict],
    *,
    n: int = config.N_QUERIES,
    seed: int = config.SEED,
    force: bool = False,
) -> tuple[dict, dict]:
    """Phase 1: one batched embedding call + one bundled entity call, cached.

    Returns ``(embeddings_by_id, entities_by_id)``; each phase file is skipped
    when it already exists (unless ``force``).
    """
    from embeddings.embedder import embedder
    from graph.entity_extraction import extract_query_entities_batch

    questions = [r["question"] for r in records]
    ids = [r["id"] for r in records]

    embeddings = None if force else persist.load("query_embeddings", n, seed)
    if embeddings is None:
        embeddings = embedder(questions)  # ONE batched Mistral call
        embeddings = {qid: emb for qid, emb in zip(ids, embeddings)}
        persist.save(embeddings, "query_embeddings", n, seed)

    entities = None if force else persist.load("query_entities", n, seed)
    if entities is None:
        extracted = extract_query_entities_batch(questions)  # ONE Gemini call
        entities = {qid: ents for qid, ents in zip(ids, extracted)}
        persist.save(entities, "query_entities", n, seed)

    return embeddings, entities


def retrieve(
    records: list[dict],
    embeddings: dict,
    entities: dict,
    *,
    top_k: int = config.TOP_K,
    candidate_k: int = config.CANDIDATE_K,
    n: int = config.N_QUERIES,
    seed: int = config.SEED,
    modes: list[config.ModeSpec] | None = None,
    force: bool = False,
    on_progress=None,
) -> dict:
    """Phase 2: run every (config x query) cell; persist final reranked chunks.

    Resumes: cells already in the cache are skipped. Each cell stores the
    post-rerank, post-graph-expansion chunk texts + graph facts — "hot and
    ready" for the generate phase. The cache is saved after every config so
    progress survives interruption.
    """
    specs = build_run_specs(modes)
    id_to_embedding = embeddings
    id_to_entities = entities

    retrievals = None if force else persist.load("retrievals", n, seed)
    retrievals = retrievals or {}
    retrievals = {
        qid: {label: cell for label, cell in cells.items() if label != "direct"}
        for qid, cells in retrievals.items()
    }

    for spec in specs:
        label = spec.label
        if label == "direct":
            continue
        done = 0
        for record in records:
            qid = record["id"]
            cells = retrievals.setdefault(qid, {})
            if label in cells:
                done += 1
                continue
            outcome = retrieve_query(
                record,
                spec.mode_name,
                use_vector=spec.use_vector,
                use_bm25=spec.use_bm25,
                use_graph=spec.use_graph,
                rerank=spec.rerank,
                top_k=top_k,
                candidate_k=candidate_k,
                query_embedding=id_to_embedding.get(qid) if spec.use_vector else None,
                query_entities=id_to_entities.get(qid) if spec.use_graph else None,
            )
            cells[label] = outcome.to_dict()
            done += 1
            if on_progress:
                on_progress(label, done, len(records))
        persist.save(retrievals, "retrievals", n, seed)

    return retrievals


def generate(
    records: list[dict],
    retrievals: dict,
    *,
    batch_size: int = config.GENERATION_BATCH_SIZE,
    n: int = config.N_QUERIES,
    seed: int = config.SEED,
    force: bool = False,
) -> dict:
    """Phase 3: bundled generation from persisted retrievals (zero retrieval work).

    ~100 pairs per config / ~40 per call => ~3 Gemini calls per config (18 RAG
    + 3 direct). Answers are persisted per (qid, label).
    """
    answers = None if force else persist.load("answers", n, seed)
    if answers is not None and answers:
        return answers

    # Add the direct cells (closed-book) to the retrieval cache shape so
    # generate_answers sees every label.
    for record in records:
        qid = record["id"]
        retrievals.setdefault(qid, {}).setdefault("direct", {})
    answers = generate_answers(records, retrievals, batch_size=batch_size)
    persist.save(answers, "answers", n, seed)
    return answers


class _CellOutcome:
    """Shim exposing the answer/retrieval view the scorer expects."""

    def __init__(self, answer: str, cell_ret: dict, context: str, generation_ms: float):
        self.answer = answer
        self.retrieved_titles = cell_ret.get("titles", [])
        self.contexts = [context] if context else []
        self.retrieval_ms = float(cell_ret.get("retrieval_ms", 0.0))
        self.generation_ms = float(generation_ms)


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


def compute_graph_lift(results: list[RunResult]) -> None:
    """In-place: F1 delta of all_three vs semantic_bm25 (same foundation)."""
    all_three = [r for r in results if r.mode == "all_three"]
    semantic_bm25 = [r for r in results if r.mode == "semantic_bm25"]
    for graph_run in all_three:
        paired = [r for r in semantic_bm25 if r.rerank == graph_run.rerank]
        if paired:
            graph_run.graph_lift = graph_run.f1 - paired[0].f1


def score(
    records: list[dict],
    retrievals: dict,
    answers: dict,
    *,
    top_k: int = config.TOP_K,
    modes: list[config.ModeSpec] | None = None,
) -> list[RunResult]:
    """Score every config from the persisted caches (no API calls)."""
    specs = build_run_specs(modes)
    results: list[RunResult] = []

    for spec in specs:
        label = spec.label
        outcomes: list[tuple[dict, object]] = []
        for record in records:
            qid = record["id"]
            cell_ans = answers.get(qid, {}).get(label, {})
            cell_ret = retrievals.get(qid, {}).get(label, {})
            answer = cell_ans.get("answer", "")
            context = cell_ans.get("context", "")
            generation_ms = float(cell_ans.get("generation_ms", 0.0))
            outcome = _CellOutcome(answer, cell_ret, context, generation_ms)
            outcomes.append((record, outcome))
        results.append(_score_run(spec, outcomes, top_k))

    compute_graph_lift(results)
    return results


def report(
    records: list[dict],
    results: list[RunResult],
    *,
    top_k: int = config.TOP_K,
    graph_ingested: bool = True,
    seed: int = config.SEED,
    stem: str | None = None,
) -> dict:
    """Phase 4: write markdown report (+ raw JSON with timings) to RESULTS_DIR."""
    from evaluation import report as report_mod

    payload = [r.to_dict() for r in results]
    meta = {
        "n": len(records),
        "seed": seed,
        "top_k": top_k,
        "graph_ingested": graph_ingested,
        "staged": True,
    }
    md_path = report_mod.write_report(
        payload, config.RESULTS_DIR, meta=meta, stem=stem
    )

    json_path = config.RESULTS_DIR / f"{md_path.stem}.json"
    json_path.write_text(
        report_mod._json_dumps(payload, meta), encoding="utf-8"
    )
    return {"md": md_path, "json": json_path, "results": payload}