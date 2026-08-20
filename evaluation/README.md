# Hybrid-RAG Evaluation Harness (HotpotQA)

Two evaluation suites for the same 7-configuration ablation matrix:

| | **staged `evaluation/`** (this doc) | **`fast_eval/`** (separate suite) |
|---|---|---|
| API key treated as | free-tier (throttled) | paid |
| Generation | bundled (~40 pairs/call) | per query (interactive) |
| Latency | intentionally `n/a` (amortized) | real per-query ms |
| Ingestion | explicit `ingest` phase | auto-checked; skipped if present |
| Concurrency | mostly sequential | `--workers` (default 2) |
| LLM-call tally | not reported | reported (rerank excluded) |

The staged suite is the **cost-efficient** one — decoupled, resumable phases that
fit free-tier API quotas; it is the suite you use when you must watch every
request. `fast_eval` is the **fast, paid** one — same ablation, interactive
per-query latency. See [`evaluation/fast_eval/README.md`](fast_eval/README.md)
for the fast suite; everything below is the staged suite.

---

A reproducible benchmark that compares retrieval strategies in Hybrid-RAG on 100
seeded HotpotQA questions and reports answer quality, retrieval quality, and
latency.

## What it compares

Four retrieval configurations, each (where applicable) with reranking **on** and
**off** — a 7-row matrix:

| Config | Vector | BM25 | Graph | Rerank | Notes |
|---|---|---|---|---|---|
| `direct` | — | — | — | n/a | Closed-book baseline: the LLM answers from parametric knowledge, no retrieval. |
| `semantic` | ✅ | | | on/off | Dense pgvector retrieval only. |
| `semantic_bm25` | ✅ | ✅ | | on/off | Dense + sparse, fused with Reciprocal Rank Fusion. |
| `all_three` | ✅ | ✅ | ✅ | on/off | Dense + sparse + **multi-hop** Kùzu graph facts. |

These map directly onto `src/retrieval/search.py::search(...)`, the explicit
(non-routing) retrieval entry point. The heuristic/agentic router is intentionally
bypassed so each path is measured in isolation.

## Staged pipeline — why the restructure

The harness runs as a **decoupled, resumable, four-phase pipeline** designed to
fit **free-tier API quotas** while staying deterministic and inspectable. The old
single-pass `run` executed the full matrix (7 configs × 100 queries = 700 cells)
sequentially, each cell doing retrieval **and** generation inline — ~800+ LLM
generation calls and ~800 embedding calls per run, infeasible on free tiers
(Groq `gpt-oss-20b` free tier allows 1K requests/day and only 200K tokens/day;
Gemini free tier is 250 RPD; Jina free reranking is 2 concurrent requests). The
restructure:

1. **Separates retrieval from generation** — retrieval and generation are now
   distinct phases, each persisted to JSON, so neither blocks the other and
   phases can be re-run independently (`--force`).
2. **Bundles API calls** — one batched embedding call for all queries, one
   bundled entity-extraction call for all queries, ~40 (question, context)
   pairs per generation call. Total API calls for n=100: **~23**.
3. **Persists "hot and ready" retrievals** — the retrieve phase stores the
   **final reranked chunks** (post-cross-encoder, post-graph-expansion) plus
   graph facts. The generate phase does *zero* retrieval work; it rebuilds
   context strings locally from the cache.
4. **Documents the latency tradeoff** — see below.

### Pipeline overview

```
prepare  →  ingest  →  artifacts  →  retrieve  →  generate  →  report
                                   (each phase resumable, cached in data/eval_cache/)
```

| Phase | Command | API calls | Output cache (`data/eval_cache/`) |
|---|---|---|---|
| Data | `prepare` | 0 (HF download) | `selected_distractor_validation_n100_seed42.json` |
| Ingestion | `ingest --with-graph` | batched NER + 1 embedding pass per doc | Postgres + Kùzu |
| Artifacts | `artifacts` | **1** Mistral embedding call + **1** bundled LLM entity call | `query_embeddings_n100_s42.json`, `query_entities_n100_s42.json` |
| Retrieval | `retrieve` | 300 Jina rerank calls (6 configs × 100, rerank half) | `retrievals_n100_s42.json` — final reranked chunks + graph facts per (query × config) |
| Generation | `generate` | **~21** bundled calls (7 configs × ~3) | `answers_n100_s42.json` |
| Report | `report` | 0 (scores from caches) | `results/eval_*.md` + `.json` |

Resume semantics: every phase skips work already present in its cache file.
`--force` recomputes a phase; `clear` (optionally `--kind <phase>`) wipes it.

### Batching design (cost efficiency)

- **Query embeddings** (`artifacts`): all 100 questions in **one** batched
  `embedder()` call (BATCH_SIZE=128), cached by query id.
- **Query entities** (`artifacts`): `extract_query_entities_batch` sends all
  100 questions in **one** bundled LLM call (`EXTRACTION__QUERY_ENTITY_MODEL`,
  default `deepseek-v4-flash`) returning `{idx: [entities]}`. Pure-LLM — no
  YAKE. A failed cell degrades to `[]` (soft-boost semantics), never an error.
- **Reranking** (`retrieve`): Jina free tier allows 2 concurrent requests;
  `RERANKER__API_RATE_LIMIT` paces request rate and a `BoundedSemaphore(2)`
  caps concurrency.
- **Generation** (`generate`): `generate_batch` (src/llm/generator.py) bundles
  ~40 (question, context) pairs per call with numbered `---Q{i}---` markers and
  parses `{idx: answer}` JSON back into order-aligned answers. Mixed RAG +
  closed-book (direct) pairs are supported in the same call. Paced by
  `GENERATOR__API_RATE_LIMIT` (2 RPM).

### Free-tier quota fit (n=100, seed 42)

| Provider | Usage | Limit |
|---|---|---|
| DeepSeek (b.ai, `deepseek-v4-flash`) | ~22 calls (1 entity + 21 gen) | generous RPM/TPD; thinking disabled (`thinking: {"type": "disabled"}`) |
| Mistral embeddings | 1 batched call | free tier |
| Jina rerank | 300 calls @ 2 concurrent | 100 RPM / 2 concurrent |

Wall time: retrieval ~30 min (Jina-paced), generation ~11 min (2 RPM).

### Latency is NOT a metric here

Per-query latency is **meaningless** in this pipeline and the report says so:

- retrieval ran in a decoupled phase (cached, resumable) — not under the same
  conditions as an interactive query;
- generation time was **amortized across bundled calls** (~40 pairs/call), so a
  per-query `generation_ms` cannot be derived from it.

Consequences:
- The markdown report renders latency columns as **`n/a`** and includes a
  `Latency caveat` block explaining this.
- Raw timings (per-cell `retrieval_ms`, per-batch `generation_ms`) are still
  preserved in `results/eval_*.json` for reference — they describe *phase*
  throughput, not interactive latency.
- Accuracy-vs-cost note: bundling is expected to cause a slight accuracy
  drop vs per-query generation (cross-query interference in long prompts);
  results should be read as "relative configuration comparison", not as
  absolute system quality.

### Ingestion batching

`ingest` is also restructured: chunk + summary embeddings are computed in
batched passes, and KG triplet extraction bundles `NER__BATCH_SIZE` (50)
chunks per call with cross-paragraph context. The graph loop remains
per-chunk resilient (one failed extraction is logged and skipped).

## Metrics

- **F1** — token-level F1 vs. the gold answer (HotpotQA's primary metric).
- **EM** — exact match after SQuAD normalization.
- **Recall** — fraction of gold-answer tokens recovered (answer recall).
- **Hit@k (top)** — supporting-fact recall@k: fraction of the gold supporting
  paragraph titles that appear in the top-k retrieved chunks. This is the "top"
  retrieval-quality metric. (`direct` has no retrieval, so it is `—`.)
- **AnsInCtx** — did any retrieved chunk actually contain the gold answer string
  (a title-alignment-free retrieval signal).
- **GraphLift** — Hit@k(all_three) − Hit@k(semantic_bm25), i.e. the contribution
  of the multi-hop knowledge graph (staged suite only).
- **Latency** — per-query wall-clock for retrieval and generation, aggregated as
  mean / median / p95. The **rerank vs non-rerank** comparison falls out of
  adjacent rows. In the staged suite latency renders `n/a` (see above); in
  `fast_eval/` it is real.

## Reproducibility

The 100 queries are chosen by a **seeded** RNG (`SEED = 42`) over the HotpotQA
`distractor` **validation** split (7,405 questions) and cached to
`evaluation/data/selected_distractor_validation_n100_seed42.json`. Re-running
reads the cache, so every run on every machine evaluates the exact same
questions. Change `--seed`/`--n` to select a different reproducible set.

Corpus ingestion is deterministic too: each paragraph is stored with
`source_id = <title>` and a stable `chunk_id = "<title>::<i>"`, so re-ingesting is
idempotent (`ON CONFLICT DO NOTHING`).

## Requirements

- The `eval` extra (the `datasets` package) for `prepare`:
  ```bash
  uv sync --extra eval
  ```
- For `ingest` and `run`, the same services the app needs: **PostgreSQL +
  pgvector**, **Ollama** (embeddings), and the configured **generator** endpoint
  (see the project `.env`).

## Usage

Run from the **repo root**:

```bash
# 1. Download HotpotQA and cache the seeded 100-query selection (needs `datasets`)
uv run --extra eval python -m evaluation.run_eval prepare

# 2. Ingest the selected paragraphs into pgvector. Add --with-graph to also build
#    the knowledge graph (required for `all_three` to contribute graph facts; slow).
uv run --extra eval python -m evaluation.run_eval ingest --with-graph

# 3. Run the full matrix and write JSON + CSV + markdown reports to evaluation/results/
uv run --extra eval python -m evaluation.run_eval run

# Or do all three at once:
uv run --extra eval python -m evaluation.run_eval all --with-graph
```

Useful flags: `--n` (query count), `--seed`, `--top-k` (final chunks scored),
`--candidate-k` (per-retriever pool before fusion/rerank).

If you ran `ingest` **without** `--with-graph`, pass `--no-graph-ingested` to
`run` so the report notes that `all_three`'s graph facts were empty.

## Latest numbers (staged, n=100, seed 42, `eval_20260819T194405Z`)

Paid-key run against the live stack (DeepSeek v4-flash generation, Mistral
embeddings, Jina reranker). Reranking improves answer quality in every
configuration, and the multi-hop graph (`all_three`) adds a measurable Hit@k
lift on top of dense+sparse fusion. Latency columns are `n/a` by design —
see ["Latency is NOT a metric here"](#latency-is-not-a-metric-here). For real
per-query latency numbers, run the `fast_eval` suite.

| Config | Rerank | F1 | EM | Recall | Hit@k (top) | AnsInCtx | GraphLift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct | no | 0.0000 | 0.0000 | 0.0000 | — | — | — |
| semantic | no | 0.2665 | 0.2100 | 0.2700 | 0.8900 | 0.8600 | — |
| semantic | yes | 0.2901 | 0.2200 | 0.2937 | 0.9450 | 0.9100 | — |
| semantic_bm25 | no | 0.2565 | 0.2000 | 0.2600 | 0.8900 | 0.8600 | — |
| semantic_bm25 | yes | 0.2901 | 0.2200 | 0.2937 | 0.9450 | 0.9100 | — |
| all_three | no | 0.2715 | 0.2200 | 0.2750 | 0.9300 | 0.9100 | 0.0150 |
| all_three | yes | 0.3028 | 0.2400 | 0.3037 | 0.9800 | 0.9500 | 0.0128 |

Accuracy notes:
- `direct` scores 0 everywhere — extractive QA without retrieval, as expected.
- **Rerank adds +0.02–0.03 F1 and +0.04–0.06 Hit@k** across the board.
- **The graph lifts Hit@k** from 0.89 (dense+sparse) / 0.945 (reranked) to
  0.93 / 0.98 — the multi-hop path does surface gold supporting facts that
  dense+sparse miss.

## Quick offline check (no live stack)

The full matrix needs PostgreSQL + Ollama + the generator. To validate the
**all-three** wiring — dense + sparse + the **multi-hop** knowledge graph, fused
and assembled into a prompt — without any of that, run the dummy-data smoke
pipeline (it stubs the dense/sparse retrievers, runs the *real* RRF fusion, the
*real* multi-hop graph BFS against a throwaway Kùzu DB, and the *real* context
builder; it stops before generation):

```bash
uv run python -m evaluation.smoke_all_three     # prints a per-signal breakdown; exits 0/1
```

The multi-hop graph traversal itself is covered by a unit test that needs no
services (it seeds a temp Kùzu graph and asserts 2-hop "bridge" facts surface
where a single hop can't):

```bash
uv run --extra dev pytest evaluation/tests/test_multihop_graph.py
```

## Outputs

`evaluation/results/eval_<UTC-timestamp>.{json,csv,md}`:
- `.json` — full metrics (incl. median latency and the retrieval/generation split) plus run metadata.
- `.csv` — flat table for spreadsheets.
- `.md` — the summary table, also printed to stdout.

## Layout

```
evaluation/
  config.py     seed, query count, dataset coords, top_k/candidate_k, mode specs
  dataset.py    HotpotQA load + seeded selection (pure `select_indices`) + caching
  corpus.py     deterministic, idempotent ingestion into pgvector (+ optional graph)
  modes.py      run a single query under one configuration (retrieval + generation)
  runner.py     orchestrate the 7-row matrix; embed questions once; score everything
  metrics.py    pure F1 / EM / recall / hit@k / answer-in-context / latency aggregation
  report.py     JSON + CSV + markdown writers
  run_eval.py   typer CLI: prepare / ingest / artifacts / retrieve / generate / report / all
  smoke_all_three.py  offline all-three (dense+sparse+multi-hop graph) smoke pipeline on dummy data
  tests/        unit tests for metrics, seeded-selector determinism, and multi-hop graph traversal
  fast_eval/    separate fast/paid suite — auto-ingest check, per-query generation, real latency
                (see fast_eval/README.md)
```

## Notes & caveats

- **Generation is non-deterministic-ish.** The generator runs at `temperature=0.1`,
  so answer metrics can vary slightly run to run even with a fixed query set;
  retrieval metrics (Hit@k, AnsInCtx) are deterministic given a fixed corpus.
- **Graph cost.** `--with-graph` runs LLM triplet extraction over the whole
  corpus and is the slow part; skip it to benchmark `semantic`/`semantic_bm25`
  (and an empty-graph `all_three`) quickly.
- **Bundling vs per-query.** Staged results come from bundled generation
  (~40 pairs/call); `fast_eval/` generates per query. Expect small absolute
  differences in the numbers.
- The harness reuses one batched question-embedding pass across all vector modes
  for a fair, fast comparison.