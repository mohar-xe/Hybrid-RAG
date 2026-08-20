# fast_eval — concurrent, paid, real-latency evaluation

A **separate** evaluation suite from the staged `evaluation/` pipeline (that one
stays for personal free-tier use — this one is untouched by it and vice versa).

differences at a glance:

| | staged `evaluation/` | `fast_eval/` |
|---|---|---|
| API key treated as | free-tier (throttled) | **paid** |
| Generation | bundled (~40 pairs/call) | **per query** (interactive) |
| Latency | intentionally `n/a` (amortized) | **real per-query** ms |
| Ingestion | explicit `ingest` phase | **auto-checked; skipped if present** |
| Concurrency | mostly sequential | `--workers` (default 2) |
| LLM-call tally | not reported | **reported** (rerank excluded) |

Same ablation matrix (7 configs × N queries):

| # | Config | vector | bm25 | graph | rerank |
|---|---|---|---|---|---|
| 1 | `direct` | – | – | – | – |
| 2 | `semantic` | ✓ | | | |
| 3 | `semantic + rerank` | ✓ | | | ✓ |
| 4 | `semantic_bm25` | ✓ | ✓ | | |
| 5 | `semantic_bm25 + rerank` | ✓ | ✓ | | ✓ |
| 6 | `all_three` | ✓ | ✓ | ✓ | |
| 7 | `all_three + rerank` | ✓ | ✓ | ✓ | ✓ |

## Latest measurements (measured 2026-08-20, b.ai `deepseek-v4-flash`, paid key)

With the paid key the per-cell latency profile is dominated by the two API
round-trips (generation + rerank), not by retrieval:

| Stage | Typical latency |
|---|---|
| Retrieval (pgvector + BM25 + graph, fused, no rerank) | **50–330 ms** (p95 ≈ 0.3 s) |
| Rerank (Jina cross-encoder) | ~1–6 s (provider round-trip; 10 req/min pacing) |
| Generation (DeepSeek v4-flash, interactive per query) | **~1–6 s** (0.9–6.2 s observed) |

Real end-to-end per-query latency (all_three + rerank, 2 workers, healthy keys):

> retrieval ≈ 0.3 s + rerank ≈ 1–6 s + generation ≈ 1–6 s → **~2.5–12 s p95‑ish**, typically a few seconds at the median with generation dominating.

Two caveats from live runs:

- The first generation call of a run cold-starts the pool and can spike to
  ~30 s; subsequent calls settle to the 1–6 s band.
- b.ai's *free/discount* key throttles sustained bursts hard (429s even at
  ~1 req/s). The suite paces entity extraction at 1 req/s and retries with
  jittered exponential backoff — a genuinely paid key removes most of this
  and pulls every stage toward the lower end of its band. Fastest observed
  per-cell composition on healthy keys: **sub-2 s** for `all_three + rerank`.

## Run it

```bash
uv run python -m evaluation.fast_eval.run
```

That's it. The suite **checks first whether the HotpotQA corpus is already
ingested** (pgvector cluster count + Kùzu chunk count) — if yes, it skips
ingestion and goes straight to retrieval. If not, it ingests (unless
`INGEST_IF_MISSING=False`), then runs.

### Options

```
--n 100          number of queries            (default: config.N_QUERIES)
--seed 42        selection seed               (default: config.SEED)
--top-k 5        final chunks per query       (default: config.TOP_K)
--candidate-k 20 candidate pool per retriever (default: config.CANDIDATE_K)
--workers 2      concurrent queries           (default: config.MAX_WORKERS)
--with-graph     ingest with Kùzu graph if missing
```

## Configuration — one file, `evaluation/fast_eval/config.py`

Everything someone needs to switch model / provider / key / workers lives there:

| Constant | Meaning |
|---|---|
| `GENERATOR_MODEL` / `GENERATOR_BASE_URL` / `GENERATOR_API_KEY` | the paid LLM (`deepseek-v4-flash` on b.ai by default; empty key → `$GENERATOR__API_KEY` → project `.env`) |
| `GENERATOR_FALLBACK_ENABLED` | off by default (paid key → no Ollama fallback pacing) |
| `GENERATOR_MAX_TOKENS` | per-query answer cap (`128`; DeepSeek v4-flash reasons by default, capping keeps answers fast — thinking is also disabled for DeepSeek/Gemini) |
| `MAX_WORKERS` | concurrency, default **2** |
| `SEED`, `N_QUERIES`, `TOP_K`, `CANDIDATE_K` | eval shape |
| `INGEST_IF_MISSING` | auto-ingest on a missing corpus (`True`), or abort (`False`) |
| `CORPUS_SOURCE_TYPE` | what the ingest-check counts (`hotpotqa`) |

Embedding (Mistral) and reranker (Jina) keys come from the project `.env`
(`EMBEDDING__*`, `RERANKER__*`), unchanged.

## What the report contains

`evaluation/fast_eval/results/fast_eval_<timestamp>.md` (+ `.json` raw cells):

- **Accuracy**: F1, EM, answer token-recall, retrieval Hit@k (top), AnsInCtx — per config.
- **Real latency (ms)**: per-config mean / median / p95 of *total* per-query
  latency, plus mean retrieval and mean generation breakdowns.
- **LLM-call tally** (rerank excluded): `llm_calls_total` counts only paid LLM
  calls — per-query **generation** + per-query **entity extraction**
  (DeepSeek v4-flash). **Embeddings** (Mistral) and **rerank** (Jina) are
  reported separately and are *not* in the total.

For `N` queries that is exactly: `7N` generation + `N` entity = `8N` LLM calls
(`N=100` → **800**), plus 1 batched embedding call and `3N` Jina rerank calls.

## Why per-query?

Rerankers and embeddings are non-LLM services (Jina / Mistral) and stay as-is.
The paid LLM is used interactively so every cell's latency is real, matching
what an end user would experience — the trade the staged suite deliberately
dropped.