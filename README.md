# Hybrid-RAG

**A production-minded hybrid retrieval-augmented generation engine that fuses dense vector search, sparse full-text search, and knowledge-graph traversal — with a reproducible benchmark harness to prove each component earns its place.**

<p>
<code>Python 3.12</code> ·
<code>PostgreSQL + pgvector</code> ·
<code>KùzuDB</code> ·
<code>Ollama</code> ·
<code>FastAPI</code> ·
<code>Typer</code> ·
<code>sentence-transformers</code> ·
<code>uv</code>
</p>

Most "RAG" projects stop at *embed → cosine search → stuff into a prompt*. This one treats retrieval as a real systems problem: a staged pipeline with bounded chunking, **coarse→fine cluster-routed dense retrieval**, sparse BM25, a deduplicated **knowledge graph**, cross-encoder reranking, optional NLI faithfulness verification, and a **seeded HotpotQA ablation matrix** that measures the contribution of every retrieval path.

---

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [How a question is answered](#how-a-question-is-answered)
- [Engineering decisions worth a look](#engineering-decisions-worth-a-look)
- [Tech stack](#tech-stack)
- [Quickstart](#quickstart)
- [CLI usage](#cli-usage)
- [HTTP API](#http-api)
- [Evaluation](#evaluation-hotpotqa-ablation-matrix)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [License](#license)

---

## Highlights

- **Three retrieval signals, fused — not bolted on.** Dense (pgvector cosine), sparse (Postgres `tsvector`/BM25), and a Kùzu knowledge graph, combined with Reciprocal Rank Fusion and re-scored by a cross-encoder.
- **Coarse→fine cluster-routed retrieval.** Chunks are clustered with spherical K-Means; each cluster keeps a *medoid*. Queries first search medoids only (coarse), route to the best clusters, then search within them (fine) — with a flat-ANN safety net and score gate so it degrades gracefully before indexing has run.
- **A knowledge graph that maintains itself.** LLM triplet extraction (two interchangeable backends), schema-validated relations, and **ANN-based near-duplicate node merging** (`hnswlib` candidate search + a lexical-overlap gate + chaining-safe "absorb-only" merges) to keep the graph clean as it grows.
- **Bring-your-own extraction model.** The default backend is a remote OpenAI-compatible API (DeepSeek); the `local` backend serves a **fine-tuned Qwen3-0.6B** (`hgr-triplet:q4`) through Ollama for fully offline triplet extraction.
- **Faithfulness verification.** An opt-in NLI cross-encoder (DeBERTa) scores how well each answer sentence is entailed by the retrieved context.
- **Evidence, not vibes.** A reproducible HotpotQA harness runs a 7-configuration ablation (direct / semantic / +BM25 / +graph × rerank on/off) and reports F1, EM, recall, retrieval hit@k, answer-in-context, and latency percentiles.
- **Two entry points over one core.** A Typer CLI (`ingest`, `reindex`, `ask`, `merge-graph`) and a FastAPI service (`/ingest`, `/query`, `/health`) with API-key auth, path-traversal protection, async ingestion, and token streaming.

---

## Architecture

```
                          ┌──────────────────────────── INGESTION ────────────────────────────┐
   PDF / YouTube / audio → │ extract → normalize (12-stage) → recursive chunk → keyphrase (YAKE)│
                          └───────────────┬───────────────────────────────┬───────────────────┘
                                          │ embed (nomic-embed-text, 256-d)│ LLM triplet extraction
                                          ▼                                ▼
                              ┌───────────────────────┐        ┌────────────────────────────┐
                              │  PostgreSQL + pgvector │        │           KùzuDB           │
                              │  chunks: HNSW + GIN    │        │  Entity ─RELATES_TO→ Entity │
                              │  cluster_id / medoid   │        │  Entity ─MENTIONED_IN→ Chunk│
                              └───────────┬────────────┘        └───────────────┬────────────┘
                                          │  reindex: K-Means + medoids          │ merge-graph: ANN dedup
                                          │                                      │
   ┌──────────────────────────────────── RETRIEVAL & ANSWER ────────────────────────────────────┐
   │ query → embed → coarse→fine cluster-routed ANN  (+BM25, +graph facts)                        │
   │       → RRF fuse → cross-encoder rerank → context builder (token budget, citations)          │
   │       → OpenAI-compatible generation (stream)   → optional NLI faithfulness score            │
   └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

Configuration, logging, a typed exception hierarchy, and idempotent schema migration sit under all of it.

## How a question is answered

The `ask` CLI command and `POST /query` share the same pipeline:

1. **Route.** A heuristic router classifies query complexity and decides whether to consult the graph.
2. **Embed.** The question is embedded with `nomic-embed-text` (truncated to 256 dims, Matryoshka-style).
3. **Retrieve (coarse→fine).** `cluster_routed_search` searches cluster medoids, selects the top clusters, then searches inside them — merging a parallel global ANN pass as a fallback and de-duplicating by best score.
4. **Rerank.** A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool down to the final top-k.
5. **Graph facts (optional).** For entity-rich queries, high-confidence relations are pulled from Kùzu.
6. **Assemble.** A token-budgeted context builder packs chunks + graph facts into a numbered, citation-ready prompt.
7. **Generate.** Any OpenAI-compatible endpoint produces the answer (streaming supported).
8. **Verify (optional).** An NLI model scores answer-vs-context faithfulness.

## Engineering decisions worth a look

These are the parts that show *why*, not just *what* — the reasoning is documented inline in the code and expanded in [`docs/report.md`](docs/report.md).

- **Bound the chunk before you embed it.** Chunking is a `RecursiveCharacterTextSplitter` (separators `["\n\n", "\n", ". ", " "]` → hard char split) packing ~300-token chunks with ~50-token overlap, using a 4-chars/token approximation so there is **no tokenizer dependency**. This replaced an earlier paragraph-embed-then-cluster design that could feed 25k-token blobs to the embedder and 400.
- **Cheap coarse pass, accurate fine pass.** Searching medoids first turns a full-corpus ANN into a small routing decision, then restricts the expensive search to a few clusters — with a similarity **score gate** that falls back to flat global ANN when clustering hasn't run or routing is weak. Retrieval works *before* `reindex`; it just gets better after.
- **The graph stays clean at scale.** Node de-duplication uses an in-memory **HNSW** index instead of O(N²) all-pairs comparison, and persists entity-name embeddings so repeat runs only embed new entities. A candidate pair must clear **both** a cosine threshold **and** a character-trigram lexical gate — embedding similarity alone conflates "same shape" with "same entity" (distinct author names or `"X theorem"` phrases score ≥0.90), so the lexical gate is what stops unrelated names from merging. Merging is then **absorb-only**: a canonical node absorbs its duplicates but a merged-away node never becomes a survivor, so a stray false match folds at most one node instead of chaining a whole component into one super-node. It is a dry-run by default; merges only happen with `--apply`.
- **Concurrency where it pays, serialization where it must.** Triplet extraction is I/O-bound, so it runs concurrently in a thread pool; Kùzu writes are serialized (a Kùzu connection isn't thread-safe, and it allows a single writer). Ingestion threads **one** graph connection through the whole phase instead of reopening per chunk.
- **Embedding is the expensive step, so don't repeat it.** Ingestion has independently toggleable `--store` and `--graph` phases; a transient extraction failure can be retried with `--no-store --graph` against already-embedded chunks, and per-chunk extraction failures are logged and skipped, never fatal.
- **Security is not an afterthought.** The API gates every endpoint behind an `X-API-Key` (constant-time compare), restricts `/ingest` to an allow-listed directory to block path traversal, and logs a loud warning if it boots unauthenticated.
- **Typed everything.** Pydantic `BaseSettings` for nested, validated config (`GROUP__FIELD` env keys, `SecretStr` for secrets); a single `BaseError` exception root; rotating file + console logging.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | **PostgreSQL + pgvector** (HNSW, cosine) | One battle-tested store for vectors *and* BM25 (`tsvector` + GIN) |
| Graph store | **KùzuDB** | Embedded, Cypher-like, no server to operate |
| Embeddings | **nomic-embed-text** via Ollama, 256-d | Local, no API cost; Matryoshka truncation shrinks the index |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** | Strong relevance signal, lazy-loaded |
| Verifier | **cross-encoder/nli-deberta-v3-base** | Entailment-based faithfulness, opt-in |
| KG extraction | **Fine-tuned Qwen3-0.6B** (local) or **DeepSeek** (remote) | Pluggable backends; offline or hosted |
| API / CLI | **FastAPI** + **Typer** | Async service and ergonomic CLI over one core |
| Tooling | **uv**, **pytest**, **Pydantic Settings** | Fast, reproducible, typed |

## Quickstart

> **Prerequisites:** [`uv`](https://docs.astral.sh/uv/), a **PostgreSQL** instance with the **pgvector** extension, and **[Ollama](https://ollama.com)** running locally for embeddings.

```bash
git clone https://github.com/yourusername/Hybrid-RAG.git
cd Hybrid-RAG

./setup.sh              # uv sync + create .env + start Ollama + pull nomic-embed-text
# ./setup.sh --local    # also download & build the fine-tuned hgr-triplet:q4 model
```

Then fill in the required values in `.env` (database credentials, a generator API key, and — for the default extraction backend — `NER__API_KEY`), create the database, and enable pgvector:

```bash
createdb hybrid_rag
psql -d hybrid_rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Run end to end against the bundled sample PDF (the package uses a **src-layout**, so CLI commands run from inside `src/`):

```bash
cd src
uv run python pipeline.py ingest ../data/sample.pdf --type pdf   # extract → chunk → embed → store (+ graph)
uv run python pipeline.py reindex                                # cluster chunks + mark medoids
uv run python pipeline.py ask "What is this document about?" --verbose
```

## CLI usage

```bash
# Ingest (two independent phases: --store/--no-store and --graph/--no-graph)
uv run python pipeline.py ingest <path|video_id> --type pdf|youtube|audio [--extractor local|deepseek]
uv run python pipeline.py ingest <path> --no-store --graph    # re-run graph only, reusing stored chunks

# Index for routed retrieval (run after ingest; safe to re-run as the corpus grows)
uv run python pipeline.py reindex

# Ask
uv run python pipeline.py ask "your question" --top-k 5 --verbose

# Deduplicate graph entities (dry run by default; --apply to commit)
uv run python pipeline.py merge-graph
uv run python pipeline.py merge-graph --apply --threshold 0.92
```

Typical order: **`ingest` → `reindex` → `ask`**, running `merge-graph --apply` any time after ingest to dedupe entities.

## HTTP API

```bash
cd src
uv run uvicorn api.app:app --reload      # http://localhost:8000  (interactive docs at /docs)
```

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | `POST` | Fire-and-forget ingestion (returns a `task_id`); file paths must resolve inside the allow-listed `API__INGEST_DIR` |
| `/ingest/{task_id}` | `GET` | Poll ingestion status |
| `/query` | `POST` | Ask a question; supports `stream: true` and optional `faithfulness` scoring |
| `/health` | `GET` | Liveness of Postgres, Ollama, and the graph store |

```bash
# Set API__API_KEY in .env, then send it on every request:
curl -X POST localhost:8000/query \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 5}'
```

> **Security:** if `API__API_KEY` is empty the service runs **unauthenticated** and warns at startup. Set it (and keep `/ingest` inside `API__INGEST_DIR`) before exposing the API to any untrusted network.

## Gradio Demo UI

A **visual demo** that exposes every stage of the retrieval pipeline — designed for technical interviews and portfolio presentations. Toggle features on/off to show the impact of each component.

```bash
cd src
uv run python demo/app.py                # http://localhost:7860
```

**Features exposed:**
- **Query routing** — shows heuristic complexity classification (simple/moderate/complex)
- **Coarse→fine retrieval** — medoid candidates (coarse) and cluster-filtered results (fine)
- **Pre/post-rerank tables** — compare candidate scores before and after cross-encoder reranking
- **Final top-K chunks** — source, score, and text preview for each retrieved chunk
- **Knowledge graph facts** — entities extracted and graph relations pulled from KùzuDB
- **Toggleable components** — enable/disable reranking and graph lookup to demonstrate ablation

**Use case:** Show hiring managers *exactly* what the system retrieves, how it routes, and why each component matters. Perfect for demonstrating systems-level thinking and retrieval architecture.

## Evaluation (HotpotQA ablation matrix)

The [`evaluation/`](evaluation/) harness benchmarks each retrieval configuration *in isolation* on a **seeded** 100-question sample of HotpotQA, so the contribution of BM25, the graph, and reranking is measurable rather than assumed.

```bash
uv sync --extra eval                                                     # adds `datasets`
uv run --extra eval python -m evaluation.run_eval prepare                # download + cache the seeded selection
uv run --extra eval python -m evaluation.run_eval ingest --with-graph    # load the corpus (graph optional, slow)
uv run --extra eval python -m evaluation.run_eval run                    # 7-config matrix → JSON/CSV/Markdown
# or: uv run --extra eval python -m evaluation.run_eval all --with-graph
```

It scores **F1**, **Exact Match**, **answer recall**, **retrieval hit@k**, **answer-in-context**, and **latency** (mean / median / p95), writing timestamped `JSON + CSV + Markdown` reports to `evaluation/results/`. Query selection is deterministic (`SEED = 42`) and the corpus is ingested with stable IDs, so runs are reproducible across machines. Pure-Python metric and selector logic is covered by unit tests:

```bash
uv run --extra dev pytest        # tests under evaluation/tests/
```

## Project structure

```
src/
├── config/          Pydantic settings + DB schema init / idempotent migrations
├── constants/       Rotating logger + typed exception hierarchy (BaseError root)
├── ingestion/       Extractor, 12-stage normalizer, recursive chunker, chunk schema
├── embeddings/      nomic-embed-text via Ollama (256-d, batched, char-capped)
├── graph/           Triplet schema, dual-backend entity extraction, ANN node merge
├── retrieval/       pgvector (cluster-routed), K-Means/medoids, reranker, Kùzu store, forced search
├── context/         Token-budgeted context builder with citations
├── llm/             OpenAI-compatible generation (streaming + closed-book)
├── reasoning/       Heuristic query router
├── verification/    NLI faithfulness scorer (opt-in)
├── api/             FastAPI service (auth, async ingest, streaming query)
├── demo/            Gradio UI for interactive demos with retrieval internals
└── pipeline.py      Typer CLI — the real entry point (ingest, reindex, ask, merge-graph)

evaluation/          Reproducible HotpotQA ablation harness (+ unit tests)
docs/report.md       Beginner-friendly, function-by-function walkthrough of the whole codebase
```

## Configuration

All settings flow through `config/settings.py` (Pydantic `BaseSettings`) and are read from a root `.env` using **nested `GROUP__FIELD` keys** (double-underscore delimiter), e.g. `DATABASE__PASSWORD`, `GENERATOR__API_KEY`, `EXTRACTION__BACKEND`, `NER__API_KEY`. Secrets use `SecretStr`; unknown keys are ignored. See [`.env.example`](.env.example) for every option with inline documentation.

Required to run: `GENERATOR__*` (generation endpoint), `DATABASE__*` (Postgres), and `NER__API_KEY` (only for the default `deepseek` extraction backend — not needed with `--extractor local`).

## Roadmap

Implemented features are listed above and verifiable in `src/`. Planned (config may be scaffolded, logic not yet wired):

- **HyDE** — embed a hypothetical answer to improve recall on sparse queries
- **RAPTOR-style hierarchical summaries** — cluster → summarize → re-cluster for broad/thematic questions
- **Graph community detection + per-community summaries**
- **Pre-generation retrieval-quality gating** — skip or re-query when context looks irrelevant
- **Agentic routing** — replace the heuristic router with a learned/tool-using planner
- **Real-time graph visualization**

## Related projects

This repo is the serving end of a pipeline: **`hgr-dataset`** (builds the triplet dataset) → **`finetuning_qwen3.5`** (fine-tunes Qwen3-0.6B → GGUF) → **Hybrid-RAG** (serves the model as the `hgr-triplet` Ollama model for KG extraction), with **`KG_triplet_evaluation`** scoring the extractor's output.

## License

MIT
