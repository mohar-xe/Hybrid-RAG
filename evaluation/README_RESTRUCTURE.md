# Evaluation restructure — staged, batched, cache-persisted

This README documents the restructuring of the evaluation harness (the
"restructure" work). It replaces the old single-pass `run` flow with a
**decoupled, resumable, four-phase pipeline** designed to fit **free-tier API
quotas** while staying deterministic and inspectable.

## Why the restructure

The old `run` command executed the full matrix (7 configs × 100 queries =
700 cells) sequentially, each cell doing retrieval **and** generation inline.
That implied ~800+ LLM generation calls and ~800 embedding calls per run —
infeasible on free tiers (Groq `gpt-oss-20b` free tier allows 1K requests/day
and only 200K tokens/day; Gemini free tier is 250 RPD; Jina free reranking is
2 concurrent requests). The restructure:

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

## Pipeline overview

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

Run everything end to end:

```bash
uv run --extra eval python -m evaluation.run_eval all --with-graph
# or stage by stage:
uv run --extra eval python -m evaluation.run_eval artifacts
uv run --extra eval python -m evaluation.run_eval retrieve
uv run --extra eval python -m evaluation.run_eval generate
uv run --extra eval python -m evaluation.run_eval report
```

Resume semantics: every phase skips work already present in its cache file.
`--force` recomputes a phase; `clear` (optionally `--kind <phase>`) wipes it.

## Batching design (cost efficiency)

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

## Free-tier quota fit (n=100, seed 42)

| Provider | Usage | Limit |
|---|---|---|
| DeepSeek (b.ai, `deepseek-v4-flash`) | ~22 calls (1 entity + 21 gen) | generous RPM/TPD; thinking disabled (`thinking: {"type": "disabled"}`) |
| Mistral embeddings | 1 batched call | free tier |
| Jina rerank | 300 calls @ 2 concurrent | 100 RPM / 2 concurrent |

Wall time: retrieval ~30 min (Jina-paced), generation ~11 min (2 RPM).

## Latency is NOT a metric here

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

## Ingestion batching (recap)

`ingest` was also restructured: chunk + summary embeddings are computed in
batched passes, and KG triplet extraction bundles `NER__BATCH_SIZE` (50)
chunks per call with cross-paragraph context. The graph loop remains
per-chunk resilient (one failed extraction is logged and skipped).

## Impact on the main application

None. `src/` never imports `evaluation/`; `search()` gained an optional
`query_entities=` kwarg (default `None` keeps demo/API/CLI behavior) and a bug
fix (structural-expansion siblings are now rehydrated into `RetrievedChunk`
objects — the old code crashed for eval callers). Reranker/generator changes
are additive pacing (rate limiters + concurrency caps).