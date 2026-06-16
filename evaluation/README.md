# Hybrid-RAG Evaluation Harness (HotpotQA)

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
| `all_three` | ✅ | ✅ | ✅ | on/off | Dense + sparse + Kùzu graph facts. |

These map directly onto `src/retrieval/search.py::search(...)`, the explicit
(non-routing) retrieval entry point. The heuristic/agentic router is intentionally
bypassed so each path is measured in isolation.

## Metrics

- **F1** — token-level F1 vs. the gold answer (HotpotQA's primary metric).
- **EM** — exact match after SQuAD normalization.
- **Recall** — fraction of gold-answer tokens recovered (answer recall).
- **Hit@k (top)** — supporting-fact recall@k: fraction of the gold supporting
  paragraph titles that appear in the top-k retrieved chunks. This is the "top"
  retrieval-quality metric. (`direct` has no retrieval, so it is `—`.)
- **AnsInCtx** — did any retrieved chunk actually contain the gold answer string
  (a title-alignment-free retrieval signal).
- **Latency** — per-query wall-clock for retrieval and generation, aggregated as
  mean / median / p95 (the table shows mean and p95). The **rerank vs non-rerank**
  comparison falls out of adjacent rows.

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
  run_eval.py   typer CLI: prepare / ingest / run / all
  tests/        unit tests for metrics + seeded-selector determinism
```

## Notes & caveats

- **Generation is non-deterministic-ish.** The generator runs at `temperature=0.1`,
  so answer metrics can vary slightly run to run even with a fixed query set;
  retrieval metrics (Hit@k, AnsInCtx) are deterministic given a fixed corpus.
- **Graph cost.** `--with-graph` runs LLM triplet extraction over the whole
  corpus and is the slow part; skip it to benchmark `semantic`/`semantic_bm25`
  (and an empty-graph `all_three`) quickly.
- The harness reuses one batched question-embedding pass across all vector modes
  for a fair, fast comparison.
