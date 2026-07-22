---
license: mit
---

# Hybrid-RAG

**A production-minded hybrid retrieval-augmented generation engine that fuses dense vector search, sparse full-text search, and knowledge-graph traversal — deployable as an API-only service with Neon Postgres.**

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

- **8-stage retrieval funnel.** Query understanding → hard SQL filters → doc-level soft ranking (summary + question ANN + entity boost) → chunk-level hybrid search (dense + lexical) → cross-list RRF fusion → cross-encoder rerank → structural graph expansion (small-to-big) → KG facts. Every stage is independently measurable.
- **Document-level clustering, not K-Means medoids.** Each document gets rich metadata at ingest time (Gemini-extracted title, summary, synthetic questions, topic tags, entities, version info). Synthetic questions are indexed individually — a query phrased as a question matches those vectors far better than a pooled centroid.
- **A knowledge graph that maintains itself.** LLM triplet extraction (two interchangeable backends), schema-validated relations, and **ANN-based near-duplicate node merging** (`hnswlib` candidate search + a lexical-overlap gate + chaining-safe "absorb-only" merges) to keep the graph clean as it grows.
- **Bring-your-own extraction model.** The default backend is a remote OpenAI-compatible API (DeepSeek); the `local` backend serves a **fine-tuned Qwen3-0.6B** (`hgr-triplet:q4`) through Ollama for fully offline triplet extraction.
- **Document versioning.** Track revisions with `--version-label` / `--supersedes` flags; the query interpreter infers `is_latest` semantics and applies `NOT EXISTS` hard filters over the version chain.
- **Faithfulness verification.** An opt-in NLI cross-encoder (DeBERTa) scores how well each answer sentence is entailed by the retrieved context.
- **Evidence, not vibes.** A reproducible HotpotQA harness runs a 7-configuration ablation (direct / semantic / +BM25 / +graph × rerank on/off) and reports F1, EM, recall, retrieval hit@k, answer-in-context, and latency percentiles.
- **Two entry points over one core.** A Typer CLI (`ingest`, `ask`, `merge-graph`) and a FastAPI service (`/ingest`, `/query`, `/health`) with API-key auth, file-upload ingestion, async processing, and token streaming.

---

## Architecture

```
                    ┌──────────────────────── INGESTION ────────────────────────┐
   PDF → │ extract → Gemini metadata (title, summary, questions, entities) │
         │                           → chunk + embed (256-d)                │
         │                           → KG triplet extraction → Kùzu         │
         └───────────────────────────┬───────────────────────────────────────┘
                                     │ document_clusters + document_questions
                                     ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  PostgreSQL + pgvector (HNSW + GIN)                      │
                    │  • document_clusters: summary_emb + synthetic questions  │
                    │  • document_questions: per-question vectors              │
                    │  • chunks: doc_id FK, dense + tsvector + keyword[]      │
                    └──────────────────────────────────────────────────────────┘
                                      +
                    ┌──────────────────────────────────────────────────────────┐
                    │  KùzuDB: Entity ─RELATES_TO→ Entity                      │
                    │          Entity ─MENTIONED_IN→ Chunk                     │
                    └──────────────────────────────────────────────────────────┘

   ┌────────────── RETRIEVAL & ANSWER (8-stage funnel) ──────────────────────────┐
   │ 0. Query understanding → semantic_query + filters                           │
   │ 1. Hard filter (SQL WHERE on doc_type, is_latest, date_after)               │
   │ 2. Doc-level soft rank (summary ANN + question ANN + entity boost, RRF)     │
   │ 3. Chunk hybrid search (dense ANN + tsvector, restricted to top docs)       │
   │ 4. Cross-list RRF fusion (doc-level + dense + lexical)                      │
   │ 5. Cross-encoder rerank (ms-marco-MiniLM)                                   │
   │ 6. Structural expansion (small-to-big graph traversal)                      │
   │ 7. Graph facts (multi-hop BFS entity context)                               │
   │ 8. Assemble + generate (citation-ready prompt → LLM)                        │
   └─────────────────────────────────────────────────────────────────────────────┘
```

Configuration, logging, a typed exception hierarchy, and idempotent schema migration sit under all of it.

## How a question is answered

The `ask` CLI command and `POST /query` share the same pipeline:

1. **Route + interpret.** The query router classifies complexity; the query interpreter (Gemini Flash-Lite or heuristic) extracts `semantic_query` + structured filters (`doc_type`, `is_latest`, `date_after`, `entities`).
2. **Hard filter.** SQL WHERE over `document_clusters` — doc_type, version-chain head, content_date — with a multi-stage safety valve that drops filters one at a time rather than returning empty results.
3. **Doc-level soft rank.** ANN over summary embeddings + max-over-question embeddings + entity/topic overlap boost, fused via RRF → top 25 documents.
4. **Chunk hybrid search.** Dense ANN + tsvector lexical search restricted to the top 25 docs → fused via RRF.
5. **Cross-list RRF fusion.** Three signals combined: doc-level rank + chunk dense rank + chunk lexical rank.
6. **Cross-encoder rerank.** `ms-marco-MiniLM-L-6-v2` re-scores the candidate pool to the final top-k.
7. **Structural expansion (optional).** Small-to-big graph traversal: seed chunks → entities → relations → sibling chunks.
8. **Graph facts (optional).** For entity-rich queries, multi-hop BFS entity context from Kùzu.
9. **Assemble.** A token-budgeted context builder packs chunks + graph facts into a numbered, citation-ready prompt.
10. **Generate.** Any OpenAI-compatible endpoint produces the answer (streaming supported).
11. **Verify (optional).** An NLI model scores answer-vs-context faithfulness.

## Engineering decisions worth a look

These are the parts that show *why*, not just *what* — the reasoning is documented inline in the code and expanded in [`docs/report.md`](docs/report.md).

- **Bound the chunk before you embed it.** Chunking is a `RecursiveCharacterTextSplitter` (separators `["\n\n", "\n", ". ", " "]` → hard char split) packing ~300-token chunks with ~50-token overlap, using a 4-chars/token approximation so there is **no tokenizer dependency**. This replaced an earlier paragraph-embed-then-cluster design that could feed 25k-token blobs to the embedder and 400.
- **Document-level clustering, not K-Means medoids.** Centroid-based document representations are lossy (they wash out the specific questions a document answers). Indexing each synthetic question as its own embedding preserves that signal — a query phrased as a question matches the question vectors far better than a pooled summary.
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
uv run python pipeline.py ask "What is this document about?" --verbose
```

## CLI usage

```bash
# Ingest (three phases: metadata + store + graph, independently toggleable)
uv run python pipeline.py ingest <path> --type pdf [--extractor local|deepseek] [--version-label v2] [--supersedes <doc_id>]
uv run python pipeline.py ingest <path> --no-store --graph    # re-run graph only, reusing stored chunks

# Ask
uv run python pipeline.py ask "your question" --top-k 5 --verbose

# Deduplicate graph entities (dry run by default; --apply to commit)
uv run python pipeline.py merge-graph
uv run python pipeline.py merge-graph --apply --threshold 0.92
```

Typical order: **`ingest` → `ask`**, running `merge-graph --apply` any time after ingest to dedupe entities.

## HTTP API

```bash
cd src
uv run uvicorn api.app:app --reload --port 8000  # http://localhost:8000  (docs at /docs)
```

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | `POST` | Ingest by server file path (gated by `API__INGEST_DIR`) |
| `/ingest/upload` | `POST` | Ingest by file upload (multipart form, PDF only) |
| `/ingest/{task_id}` | `GET` | Poll ingestion task status |
| `/query` | `POST` | Ask a question; supports `stream: true` and optional `faithfulness` |
| `/health` | `GET` | Liveness of Postgres and graph store |

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
- **Query routing + filters** — complexity classification and interpreted filters (doc_type, is_latest, entities)
- **Doc-level soft rank** — summary + question ANN scores for the top candidate documents
- **Pre/post-rerank tables** — compare candidate chunks before and after cross-encoder reranking
- **Structural expansion** — sibling chunks surfaced via small-to-big graph traversal
- **Knowledge graph facts** — entities extracted and graph relations pulled from KùzuDB
- **Toggleable components** — enable/disable reranking, graph facts, and structural expansion to demonstrate ablation

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
├── ingestion/       Extractor, document_cluster (Gemini metadata), recursive chunker, chunk schema
├── embeddings/      nomic-embed-text via Ollama / API / sentence-transformers (256-d)
├── graph/           Triplet schema, dual-backend entity extraction, ANN node merge
├── retrieval/       pgvector (8-stage funnel: hard_filter_docs, doc_level_soft_rank, document_routed_search),
│                      reranker, Kùzu store (structural_expansion + multi-hop entity context)
├── context/         Token-budgeted context builder with citations
├── llm/             OpenAI-compatible generation (streaming + closed-book)
├── reasoning/       Query router + query interpreter (Gemini Flash-Lite → semantic_query + filters)
├── verification/    NLI faithfulness scorer (opt-in)
├── api/             FastAPI service (auth, file-upload ingest, async query)
├── cache.py         LRU cache (embedding, metadata, query)
├── models/          Client, fallback, rate limiter
├── demo/            Gradio UI for interactive demos with retrieval internals
└── pipeline.py      Typer CLI — the real entry point (ingest, ask, merge-graph)

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

## Deploy to Render (free tier, includes Postgres)

This project runs as a **fully API-only service** (no local models) with a managed Postgres database. Deploy to [Render's free tier](https://render.com/pricing) — includes Postgres, Docker support, and auto-deploys from GitHub. Note: free web services spin down after 15 minutes of inactivity (cold start ~30s).

### Option A: One-click with Render Blueprint (recommended)

The included [`render.yaml`](render.yaml) declares both the web service and a free Postgres database. Render auto-detects it:

1. Push this repo to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** → **Blueprint**.
3. Connect your GitHub repo. Render reads `render.yaml` and creates:
   - A **Postgres database** (`hybrid-rag-db`, free tier, Singapore)
   - A **Web Service** (`hybrid-rag`, Docker, free tier, Singapore)
4. The database connection string is auto-wired via `DATABASE__CONNECTION_STRING`. No manual DB config needed.
5. Enable pgvector — connect to your database (Render dashboard → Postgres → Connect → PSQL command) and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
6. The Blueprint pre-configures the following. For `sync: false` items, click the pencil to fill in your values:

| Variable | What to put |
|----------|-------------|
| `GENERATOR__BASE_URL` | `https://openrouter.ai/api/v1` |
| `GENERATOR__MODEL` | `google/gemini-3.6-flash` |
| `GENERATOR__API_KEY` | [Get from OpenRouter](https://openrouter.ai/keys) |
| `NER__BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `NER__MODEL` | `gemini-3.6-flash` |
| `NER__API_KEY` | [Get from Google AI Studio](https://makersuite.google.com/app/apikey) |
| `EMBEDDING__API_BASE_URL` | `https://api.mistral.ai/v1` |
| `EMBEDDING__API_KEY` | [Get from Mistral](https://console.mistral.ai/api-keys/) |
| `RERANKER__API_BASE_URL` | `https://api.mistral.ai/v1` |
| `RERANKER__API_KEY` | Same Mistral key |

   All local-model fallbacks are **disabled in the blueprint** (`*__FALLBACK_ENABLED=false`) — the deployment is API-only. If a primary API fails, you get a clean error rather than a connection-refused to a non-existent Ollama or missing spaCy.

7. Go to your **Web Service** → **Manual Deploy** → **Deploy latest commit**. Wait for **Live** (~3 min).

Your API is live at `https://hybrid-rag.onrender.com`.

### Option B: Manual setup (without Blueprint)

1. Push this repo to GitHub.
2. In Render Dashboard → **New +** → **Web Service**, connect your repo.
3. Settings:
   - **Name:** `hybrid-rag`
   - **Environment:** `Docker`
   - **Plan:** Free
   - **Region:** Singapore
4. Add a **Postgres database**: Render Dashboard → **New +** → **PostgreSQL** (free tier, Singapore).
5. Enable pgvector — connect to your DB and run: `CREATE EXTENSION IF NOT EXISTS vector;`
6. Copy the **Internal Connection String** from your database dashboard and set these env vars on your web service:

```
DATABASE__CONNECTION_STRING=<paste-internal-connection-string>
GENERATOR__BASE_URL=https://openrouter.ai/api/v1
GENERATOR__MODEL=google/gemini-3.6-flash
GENERATOR__API_KEY=<your-key>
NER__BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
NER__MODEL=gemini-3.6-flash
NER__API_KEY=<your-key>
EMBEDDING__BACKEND=api
EMBEDDING__API_BASE_URL=https://api.mistral.ai/v1
EMBEDDING__API_MODEL=mistral-embed
EMBEDDING__API_KEY=<your-key>
RERANKER__BACKEND=api
RERANKER__API_BASE_URL=https://api.mistral.ai/v1
RERANKER__API_MODEL=mistral-large-latest
RERANKER__API_KEY=<your-key>
API__API_KEY=<your-key>
```

   Also add these to **disable local fallbacks** (no Ollama or spaCy in the container):

```
EXTRACTION__FALLBACK_ENABLED=false
NER__FALLBACK_ENABLED=false
EMBEDDING__FALLBACK_ENABLED=false
GENERATOR__FALLBACK_ENABLED=false
RERANKER__FALLBACK_ENABLED=false
METADATA__FALLBACK_ENABLED=false
METADATA__FALLBACK_BACKEND=none
VERIFIER__FALLBACK_ENABLED=false
```

7. Deploy. Render builds and starts the service.

### Ingest & query

```bash
# Upload a PDF
curl -X POST https://hybrid-rag.onrender.com/ingest/upload \
  -H "X-API-Key: $API_KEY" \
  -F "file=@document.pdf"

# Poll status
curl https://hybrid-rag.onrender.com/ingest/<task_id> \
  -H "X-API-Key: $API_KEY"

# Ask
curl -X POST https://hybrid-rag.onrender.com/query \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 5}'
```

Interactive docs at `https://hybrid-rag.onrender.com/docs`.

### Limitations

- **Free tier spin-down.** Render spins down a free web service after 15 min of inactivity. First request after idle takes ~30s (cold start). Upgrade to a paid plan for zero spin-down.
- **Kùzu graph store** is file-based and resets on each deploy (no persistent disk on Render free tier). Acceptable for demos. For production, store graph data in Postgres as JSONB.
- **Uploaded files** stored in `/tmp/` are lost on restart. Re-ingest after redeploy.

## Related projects

This repo is the serving end of a pipeline: **`hgr-dataset`** (builds the triplet dataset) → **`finetuning_qwen3.5`** (fine-tunes Qwen3-0.6B → GGUF) → **Hybrid-RAG** (serves the model as the `hgr-triplet` Ollama model for KG extraction), with **`KG_triplet_evaluation`** scoring the extractor's output.

## License

MIT
