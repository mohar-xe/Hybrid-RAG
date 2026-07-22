# AGENTS.md — Hybrid-RAG

Guidance for AI coding agents working in this repository. Read this before making changes.

## What this project is

A hybrid retrieval-augmented generation system that combines **document-level clustering**
with **knowledge-graph traversal (KùzuDB)** for question answering. Python 3.12+, managed
with `uv`.

The README separates **"What Exists"** from **"What Will Be Implemented"**, but it has drifted
out of date — it still lists retrieval, the graph store, entity extraction, the CLI, and the
FastAPI app as "planned" even though they are implemented. **Treat `src/` as the source of
truth for what exists**, not the README. Do not assume a planned feature exists — verify in
`src/` first.

**Document-level clustering (8-stage retrieval funnel) has replaced the old K-Means/medoid
path.** The old `cluster_routed_search`, `vector_search`, `bm25_search`, `hybrid_search`
functions are kept for backward compatibility (eval harness uses them), but `ask`/`/query`
now use the new pipeline. `retrieval/cluster.py`, the `reindex` CLI command, and medoid DDL
creation in `init_db.py` have been removed.

## Setup

Frictionless path for a fresh clone:

```bash
./setup.sh            # uv sync + .env + Ollama running + pull nomic-embed-text
./setup.sh --local    # also download the fine-tuned GGUF from HF and build hgr-triplet:q4
```

Or manually:

```bash
uv sync
cp .env.example .env   # then fill in REQUIRED values
```

`setup.sh --local` downloads `qwen3-0.6b.F16.gguf` from the HF repo
`mohar07/qwen3-0.6b-kg-triplets` into `model/` and runs `ollama create hgr-triplet:q4 -f
model/Modfile`. Set `HF_TOKEN` in `.env` for faster/authenticated downloads (optional; the repo
is public, and the app itself ignores `HF_TOKEN` — only `setup.sh` reads it). The GGUF is **not**
committed to the repo; it must be pulled.

External services this project talks to (must be running for end-to-end use):
- **PostgreSQL with the `pgvector` extension** (chunk storage + document clusters + HNSW/GIN indexes).
- **Ollama** at `http://localhost:11434` for embeddings (`nomic-embed-text`) and, when the
  `local` extraction backend is selected, KG extraction via the fine-tuned `hgr-triplet:q4`
  model (produced by the sibling `finetuning_qwen3.5` project; `model/` holds the GGUF +
  an Ollama `Modfile`).
- **Remote OpenAI-compatible APIs** for the default backends (Gemini for NER/metadata,
  Mistral for embeddings/reranker, OpenRouter for generation). See "Entity-extraction backends".

The default NER backend has changed from DeepSeek to Gemini
(`https://generativelanguage.googleapis.com/v1beta/openai`, model `gemini-3.6-flash`)
to match the deployment config. The `EXTRACTION__BACKEND=deepseek` still refers to
whatever `NER__*` settings point at — the naming is historical.

## Running

The package uses a **src-layout with top-level intra-package imports** (e.g.
`from constants.logger import setup_logger`, `from ingestion.extractor import Extractor`).
That means code runs with **`src/` as the import root**. Run the CLI from inside `src/`:

```bash
cd src
uv run python pipeline.py ingest <path> --type pdf [--extractor local|deepseek] [--version-label v2] [--supersedes <doc_id>]
uv run python pipeline.py ask "<question>" --verbose
uv run python pipeline.py merge-graph --apply   # deduplicate graph entities
```

Typical end-to-end order: **`ingest` (with version flags) → `ask` → `merge-graph`** (any time after ingest).

`ingest` has **three independent phases**, separately toggleable:
- **metadata** (always runs with store): Gemini/spaCy doc-level extraction → `document_clusters` + `document_questions`.
- **store** (`--store`/`--no-store`): chunk → embed → pgvector. The slow part (CPU embedding).
- **graph** (`--graph`/`--no-graph`): triplet extraction → Kùzu.

Re-run just the graph phase against already-stored chunks: `ingest <path> --no-store --graph`.
The graph loop is **per-chunk resilient** — one failed extraction is logged and skipped, not
fatal.

Document versioning: `--version-label "v3" --supersedes <doc_id>` sets up a version chain.
The detection chain (CLI → Gemini → spaCy → filename heuristic) resolves `version_info`;
`--version-label` overrides all.

- The **root `main.py` is a placeholder stub** — the real entry point is `src/pipeline.py`
  (a Typer app with `ingest`, `ask`, `merge-graph` commands; `reindex` has been removed).
- `merge-graph` is destructive only with `--apply`; default is a dry run. Preserve that
  dry-run-by-default behavior. `--threshold/-s` overrides
  `GRAPH__MERGE_SIMILARITY_THRESHOLD` (default `0.90`) for a single run.

To run the FastAPI app instead of the CLI (also from inside `src/`):

```bash
cd src
uv run uvicorn api.app:app --reload      # http://localhost:8000  (docs at /docs)
```

To run the **Gradio demo UI** (exposes retrieval internals for technical demos):

```bash
cd src
uv run python demo/app.py                # http://localhost:7860
```

The Gradio demo is designed for **interview/portfolio use** — it shows every stage of the
8-stage retrieval funnel (hard filter → doc soft rank → chunk hybrid → RRF fusion → rerank →
graph expansion → facts → generate) with toggleable features (reranking, knowledge graph,
structural expansion).

## Architecture / layout (`src/`)

```
config/      Pydantic settings (settings.py) + DB schema init/migrations (init_db.py)
constants/   logger.py (rotating file + console) + exceptions.py (typed hierarchy)
ingestion/   extractor (PDF-only), normalize, chunker, chunk_schema, document_cluster (Gemini metadata)
embeddings/  embedder.py — nomic-embed-text via Ollama / API / sentence-transformers, 256-dim
graph/       entity_extraction.py, schema.py, merge.py    [implemented]
retrieval/   pgvector.py (8-stage funnel: hard_filter_docs, doc_level_soft_rank, document_routed_search),
             reranker.py (cross-encoder), kuzu_store.py (structural_expansion, get_entity_context)
             cluster.py [REMOVED — replaced by document-level clustering]
reasoning/   router.py (complexity), query_interpreter.py (semantic_query + filters)
context/     builder.py — context assembly + citations
llm/         generator.py — OpenAI-compatible generation
verification/ verifier.py — NLI faithfulness scoring [planned — settings exist]
api/         app.py — FastAPI surface (/ingest, /query, /health)
demo/        app.py — Gradio UI for interactive demos with retrieval internals
pipeline.py  Typer CLI (ingest, ask, merge-graph) — real entry point
cache.py     LRU cache with 3 shared instances (embedding, metadata, query)
models/      client.py, fallback.py, rate_limiter.py — shared infrastructure
```

## Chunking & embedding (ingestion)

`ingestion/chunker.py` is a **RecursiveCharacterTextSplitter** (`chunk_text`): it splits on a
separator priority list `["\n\n", "\n", ". ", " "]`, descending to the next separator for any
piece still over budget and hard-splitting on characters as a last resort (`_atomic_splits`),
then greedily re-packs pieces with a sliding-window overlap (`_merge_splits`). Sizes are
**character-approximated tokens** (`CHARS_PER_TOKEN = 4`): `CHUNK_CHARS = 1200` (~300 tokens),
`OVERLAP_CHARS = 200` (~50 tokens).

`chunk_enrich` now accepts a `doc_id: str | None` kwarg to link chunks to their parent document.

`embeddings/embedder.py` (`nomic-embed-text`, 2048-token context):
- Each input is capped to `MAX_INPUT_CHARS = 8000` client-side and `truncate=True` is passed to
  Ollama — a defence-in-depth against over-long inputs.
- `BATCH_SIZE = 16` (Ollama), `128` (API/sentence-transformers).
- Output is truncated to 256 dims (Matryoshka).
- `normalize.remove_control_chars` removes NUL/control bytes that PostgreSQL `text` rejects.

## Retrieval pipeline (8-stage funnel)

The old K-Means/medoid cluster-routed retrieval is **replaced** by document-level clustering.
`ask`/`/query` run this funnel:

1. **QUERY UNDERSTANDING** (`reasoning/query_interpreter.py`): Gemini Flash-Lite or heuristic →
   `semantic_query` + `filters` (doc_type, is_latest, date_after, entities).
2. **HARD FILTER** (`retrieval/pgvector.hard_filter_docs`): SQL WHERE over `document_clusters`
   (doc_type, supersedes chain, content_date). 4-stage safety valve drops filters one at a time
   if results are empty.
3. **DOC-LEVEL SOFT RANK** (`retrieval/pgvector.doc_level_soft_rank`): ANN over summary
   embeddings + question embeddings (max per doc) + entity/topic overlap boost → RRF fusion →
   top 25 docs.
4. **CHUNK-LEVEL HYBRID SEARCH** (`document_routed_search`): Dense ANN + lexical (tsvector)
   restricted to the top 25 docs → RRF fusion of (doc-level rank + dense rank + lexical rank).
5. **CROSS-ENCODER RERANK**: `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-k (default 5).
6. **STRUCTURAL EXPANSION** (`retrieval/kuzu_store.structural_expansion`): Small-to-big graph
   traversal from reranked chunks: chunk → entities → relations → sibling chunks.
7. **GRAPH FACTS** (`get_entity_context`): Multi-hop BFS entity context for the query.
8. **ASSEMBLE + GENERATE**: Title/summary + expanded chunks + graph facts → LLM.

The old `cluster_routed_search`, `vector_search`, `bm25_search`, `hybrid_search` functions are
still in `pgvector.py` for backward compat (eval harness), but are not called by the main pipeline.

## Graph store (Kùzu)

`retrieval/kuzu_store.py` defines four tables: `Entity` (name PK, entity_type), `Chunk`
(chunk_id PK, text, source_id), `RELATES_TO` (Entity→Entity, relation_type, weight), and
`MENTIONED_IN` (Entity→Chunk).

Key notes:
- `upsert_triplets` MERGEs entities and their `RELATES_TO` edge.
- `link_entities_to_chunk` **MERGEs the `Chunk` node itself** before creating `MENTIONED_IN`
  edges — if you drop that MERGE the edges silently never form.
- `get_entity_context` does **multi-hop** BFS over `RELATES_TO`, bidirectionally, up to
  `GRAPH__MAX_HOPS` (default 2), with `GRAPH__MIN_RELATION_WEIGHT` filtering,
  `GRAPH__PER_HOP_NEIGHBORS` fan-out cap, and `GRAPH__MAX_FACTS` total cap.
- `structural_expansion(chunk_ids)` implements small-to-big: seed chunks → entities → related
  entities → sibling chunks. This expands the *chunk pool* rather than adding peripheral facts.
- `get_connection()` returns `(db, conn)`. Every graph function takes an optional keyword-only
  `conn=` and opens its own only when none is passed. Ingestion opens one connection up front
  and threads it through every per-chunk call — follow this pattern.

## Entity-extraction backends

KG triplet extraction has **two interchangeable backends**, in
`src/graph/entity_extraction.py`:

- **`deepseek` (default)** — remote OpenAI-compatible endpoint.
- **`local`** — Ollama fine-tuned `hgr-triplet:q4`.

`extract_entities(text, backend=...)` dispatches between them; `backend=None` falls back to
`EXTRACTION__BACKEND` (default `deepseek`). Select per run with `--extractor/-e` or
`/ingest?extractor=`. Both backends share one system prompt, JSON parsing (`_parse_json` —
returns `[]` for empty responses), and an idempotent schema check. A third backend goes in the
`_BACKENDS` dict.

Concurrency: `extract_entities_batch` runs I/O-bound calls via `ThreadPoolExecutor`
(order-preserved), default concurrency 10. The ingest graph phase writes results to Kùzu
serially (Kùzu connection is not thread-safe; the API uses `_GRAPH_WRITE_LOCK`).

`NER__DISABLE_THINKING` (default `true`) disables DeepSeek's reasoning — critical for
structured extraction. Leave this on unless a backend rejects the field.

## Document-level clustering (metadata extraction)

`ingestion/document_cluster.py` replaces the old K-Means/medoid approach:

- `extract_document_metadata(text, source_id)`: Gemini 3.6 Flash → JSON with title, summary,
  synthetic_questions, doc_type, topic_tags, entities, content_date, version_info. Falls back
  to spaCy heuristics. Documents >1M chars skip Gemini entirely. On total failure returns
  minimal dict (data-quality gradient).
- `create_document_cluster(doc_id, source_id, source_type, metadata, text, ...)`: Embeds
  summary + questions, writes `document_clusters` + `document_questions` rows. Accepts
  `supersedes_doc_id`, `is_versioned`, `version_label` kwargs for version chains.

Version detection chain (first non-None wins): CLI `--version-label` → Gemini `version_info`
→ spaCy regex → filename heuristic. The `NOT EXISTS` query pattern identifies the chain head
for `is_latest` semantics.

## Caching

`src/cache.py` defines an LRU cache with three pre-instantiated shared instances:
- `embedding_cache`: single-string embedding results (keyed by MD5 of text).
- `metadata_cache`: document metadata (keyed by MD5 of first 500 chars + source_id).
- `query_cache`: query interpretation results (keyed by MD5 of normalized question).

Integrated in `embeddings/embedder.py`, `ingestion/document_cluster.py`,
`reasoning/query_interpreter.py`.

## Docker deployment

API-only deployment. No local models (sentence-transformers, spaCy, Ollama). All defaults
use remote APIs. See `Dockerfile`, `docker-compose.yml`, `.dockerignore`.

**All fallbacks are disabled in deployment.** Every component (NER, extraction, embedding,
generator, reranker, metadata, verifier) has `*__FALLBACK_ENABLED=false` set in `render.yaml`.
When a primary API fails, the error propagates cleanly rather than trying to hit a non-existent
Ollama at `http://localhost:11434` or import an uninstalled spaCy. Settings.py defaults keep
`fallback_enabled=True` for local dev.

## Conventions (follow these)

- **Configuration:** all settings go through `config/settings.py` (Pydantic `BaseSettings`).
  The root `.env` uses **nested `GROUP__FIELD` keys** with a double-underscore delimiter
  (e.g. `DATABASE__PASSWORD`, `GENERATOR__API_KEY`, `METADATA__BASE_URL`). Secrets use
  `SecretStr`. Unknown keys are ignored (`extra="ignore"`). Get config via `get_settings()`.
  `ExtractionSettings`, `NERSettings`, `MetadataSettings`, `QuerySettings`,
  `EmbeddingSettings` are also independently instantiable.
- **Postgres connections:** build the libpq string via `settings.database.conninfo`.
  Don't hand-assemble `host=... port=...` strings.
- **Graph tuning:** `GraphSettings` (`GRAPH__*`) holds `db_path`, `merge_similarity_threshold`,
  `min_relation_weight`, `max_hops`, `max_facts`, `per_hop_neighbors`. Route graph behavior
  through these rather than hard-coding constants.
- **Logging:** obtain loggers with `setup_logger(__name__)` from `constants.logger`. Logs
  go to `logs/app.log` (rotating, 5 MB × 3) plus stdout. Don't use bare `print` for
  diagnostics in library code (the CLI uses `typer.echo` for user-facing output).
- **Errors:** raise the typed exceptions in `constants/exceptions.py` (e.g.
  `ConfigurationError`, `TextExtractionError`, `GraphError`) rather than bare `Exception`.
  Every project exception derives from a single `BaseError` root.
- **Imports:** keep the top-level style (`from graph.merge import ...`), consistent with
  the rest of the package and the editable install (`src/` is the package root).
- Heavy/optional imports inside CLI commands are imported lazily (inside the function) on
  purpose — keep that pattern so `--help` stays fast and optional paths don't force imports.
- **Filters:** `doc_type` and `is_latest` get hard SQL WHERE. Entity/topic overlap is a soft
  RRF boost, not a hard gate — avoiding false negatives from inexact naming.
- **Safety valve:** `hard_filter_docs` drops filters one at a time when results are empty.
  A parser mistake should never produce "no results found."

## Security notes

- Never commit a populated `.env` (it is gitignored). Secrets belong in `.env` only.
- `src/api/app.py` is a **local-first FastAPI app with optional API key auth**. Configure
  `API__KEY` to gate endpoints; without it the app logs a loud warning at startup. Ingestion
  runs arbitrary file paths gated under `API__INGEST_DIR` — do not expose to a public network
  without authentication and input validation.

## Verification

There is **no test suite or linter configured** yet. After changes:
- Confirm the module imports cleanly from `src/` (e.g. `cd src && uv run python -c "import pipeline"`).
  Several modules call `get_settings()` at **import time**, so importing requires a populated
  `.env` (at minimum the `DATABASE__*` and `GENERATOR__*` required fields).
- For ingestion/retrieval changes, a manual `ingest` then `ask` against a sample file in
  `data/` is the realistic smoke test (requires Postgres + Ollama running).
- If you add tests, place them under a top-level `tests/` and prefer `pytest`.

## Related projects (same workspace)

This repo is the consumer end of a pipeline:
`hgr-dataset` (builds the triplet dataset) → `finetuning_qwen3.5` (fine-tunes Qwen3-0.6B →
GGUF) → **Hybrid-RAG** (serves the model as the `hgr-triplet` Ollama model for extraction).
`KG_triplet_evaluation` scores that model's triplet output.
