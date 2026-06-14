# AGENTS.md — Hybrid-RAG

Guidance for AI coding agents working in this repository. Read this before making changes.

## What this project is

A hybrid retrieval-augmented generation system that combines **vector search (pgvector)**
with **knowledge-graph traversal (KùzuDB)** for question answering. Python 3.12+, managed
with `uv`.

The README separates **"What Exists"** from **"What Will Be Implemented"**, but it has drifted
out of date — it still lists retrieval, the graph store, entity extraction, the CLI, and the
FastAPI app as "planned" even though they are implemented. **Treat `src/` as the source of
truth for what exists**, not the README. Do not assume a planned feature exists — verify in
`src/` first.

- **Implemented:** ingestion (PDF/YouTube/audio extraction → normalize → hierarchical chunk →
  keyphrase → embed), pgvector storage + schema init, config/logging/exception infrastructure,
  Ollama embeddings, LLM entity/triplet extraction (local + remote backends), Kùzu graph store,
  near-duplicate node merge, hybrid retrieval (vector + BM25), Kùzu graph-context lookup,
  heuristic query router, context assembly + citations, OpenAI-compatible generation, and a
  FastAPI app (`/ingest` async + status, `/query`, `/health`).
- **Planned / scaffolded (config may exist but logic is not wired yet):** cross-encoder
  reranking (`RerankerSettings` + the router's `use_reranker` flag exist, but no reranker is
  actually invoked), NLI self-verification (`VerifierSettings` exists; `/query` returns
  `faithfulness: None`), retrieval quality scoring, HyDE, RAPTOR summaries, graph community
  summaries, streaming `/query` responses (the `stream` flag is accepted but ignored), real-time
  graph visualization, and the evaluation suite.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in REQUIRED values
```

External services this project talks to (must be running for end-to-end use):
- **PostgreSQL with the `pgvector` extension** (chunk storage + HNSW/GIN indexes).
- **Ollama** at `http://localhost:11434` for embeddings (`nomic-embed-text`) and, when the
  `local` extraction backend is selected, KG extraction via the fine-tuned `hgr-triplet:q4`
  model (produced by the sibling `finetuning_qwen3.5` project; `model/` holds the GGUF +
  an Ollama `Modfile`).
- **A remote OpenAI-compatible API** for the default `deepseek` extraction backend
  (DeepSeek V4 Flash). Requires `NER__API_KEY` in `.env`. See "Entity-extraction backends".

## Running

The package uses a **src-layout with top-level intra-package imports** (e.g.
`from constants.logger import setup_logger`, `from ingestion.extractor import Extractor`).
That means code runs with **`src/` as the import root**. Run the CLI from inside `src/`:

```bash
cd src
uv run python pipeline.py ingest <path> --type pdf|youtube|audio [--extractor local|deepseek]
uv run python pipeline.py ask "<question>" --top-k 5 --verbose
uv run python pipeline.py merge-graph            # dry run: lists candidates
uv run python pipeline.py merge-graph --apply    # actually merges
uv run python pipeline.py merge-graph --threshold 0.95   # override similarity cutoff
```

- The **root `main.py` is a placeholder stub** — the real entry point is `src/pipeline.py`
  (a Typer app with `ingest`, `ask`, `merge-graph` commands).
- `merge-graph` is destructive only with `--apply`; default is a dry run. Preserve that
  dry-run-by-default behavior. `--threshold/-s` overrides
  `GRAPH__MERGE_SIMILARITY_THRESHOLD` (default `0.90`) for a single run.

To run the FastAPI app instead of the CLI (also from inside `src/`):

```bash
cd src
uv run uvicorn api.app:app --reload      # http://localhost:8000  (docs at /docs)
```

## Architecture / layout (`src/`)

```
config/      Pydantic settings (settings.py) + DB schema init (init_db.py)
constants/   logger.py (rotating file + console) + exceptions.py (typed hierarchy)
ingestion/   extractor, normalize, chunker, chunk_schema  [implemented]
embeddings/  embedder.py — nomic-embed-text via Ollama, 256-dim (Matryoshka)
graph/       entity_extraction.py, schema.py, merge.py    [implemented]
retrieval/   pgvector.py (vector + BM25 hybrid), kuzu_store.py [implemented]
context/     builder.py — context assembly + citations    [implemented]
llm/         generator.py — OpenAI-compatible generation   [implemented]
reasoning/   router.py — heuristic query routing [implemented]; verifier [planned]
api/         app.py — FastAPI surface (/ingest, /query, /health) [implemented core]
pipeline.py  Typer CLI (real entry point)
```

## Retrieval routing

`reasoning/router.py` is a **heuristic (no-LLM) router**: `classify_query` labels a query
`simple | moderate | complex` from word count, leading question phrase, capitalized-token
("entity") count, and comparison/multi-hop keywords. `route_retrieval` maps that label to a
strategy dict (`use_vector`, `use_bm25`, `use_graph`, `use_reranker`, `top_k`). Both the CLI
`ask` and the API `/query` consume this dict:
- `use_bm25` → `retrieval.pgvector.hybrid_search`, else `vector_search` on a fresh embedding.
- `use_graph` → pull capitalized tokens as candidate entities and fetch
  `retrieval.kuzu_store.get_entity_context`.
- `use_reranker` is **set but not yet acted on** — no reranker runs. If you wire one up, read
  `RerankerSettings` and gate it on this flag rather than adding a new toggle.

## Graph store (Kùzu)

`retrieval/kuzu_store.py` defines four tables: `Entity` (name PK, entity_type), `Chunk`
(chunk_id PK, text, source_id), `RELATES_TO` (Entity→Entity, relation_type, weight), and
`MENTIONED_IN` (Entity→Chunk). Notes for anyone touching this:
- `upsert_triplets` MERGEs entities and their `RELATES_TO` edge.
- `link_entities_to_chunk(entity_names, chunk_id, text, source_id)` **MERGEs the `Chunk` node
  itself** before creating `MENTIONED_IN` edges — nothing else inserts `Chunk` nodes, so if you
  drop that MERGE the edges silently never form (the entity→chunk `MATCH` finds no chunk). Pass
  `text`/`source_id` so the node is populated on creation.
- `get_entity_context` only traverses `RELATES_TO`, filtered by `GRAPH__MIN_RELATION_WEIGHT`.
- `get_connection()` returns `(db, conn)`. Every graph function
  (`init_graph_schema`, `upsert_triplets`, `link_entities_to_chunk`, `get_entity_context`)
  takes an optional keyword-only `conn=` and opens its own only when none is passed. Ingestion
  (`pipeline.ingest` and the API's `_run_ingestion`) opens **one** connection up front and
  threads it through every per-chunk call instead of reopening the DB in the loop — follow this
  pattern for any multi-statement graph work, and keep the `db` handle alive in a local (e.g.
  `graph_db, graph_conn = get_connection()`) so the `Database` isn't GC'd while the connection
  is in use.

## Entity-extraction backends

KG triplet extraction has **two interchangeable backends**, in
`src/graph/entity_extraction.py`:

- **`deepseek` (default)** — `extract_entities_api`: a remote OpenAI-compatible endpoint
  (`POST {NER__BASE_URL}/chat/completions`), defaulting to DeepSeek V4 Flash. Configured by
  `NERSettings` (`NER__*` env keys); requires the `NER__API_KEY` secret and raises
  `ConfigurationError` if it is missing.
- **`local`** — `extract_entities_local`: the Ollama fine-tuned `hgr-triplet:q4` model via
  `/api/chat`. Configured by `ExtractionSettings` (`EXTRACTION__*` env keys).

`extract_entities(text, backend=...)` dispatches between them; `backend=None` falls back to
`EXTRACTION__BACKEND` (default `deepseek`). Select per run with the CLI `--extractor/-e`
(`local|deepseek`) or the `POST /ingest?extractor=` query param. `extract_entities_llm`
stays as a backward-compatible alias for the local path. Both backends share one system
prompt (`_build_system_prompt`), JSON parsing (`_parse_json`), and an **idempotent schema
check** (`validate_triplets` — validates raw dicts against the `Triplet` schema, drops invalid
ones, and passes through items that are already validated `Triplet` instances). Keep that
shared path intact, and register any third backend in `_BACKENDS`.

## Conventions (follow these)

- **Configuration:** all settings go through `config/settings.py` (Pydantic `BaseSettings`).
  The root `.env` uses **nested `GROUP__FIELD` keys** with a double-underscore delimiter
  (e.g. `DATABASE__PASSWORD`, `GENERATOR__API_KEY`, `EXTRACTION__BACKEND`, `NER__API_KEY`).
  Secrets use `SecretStr`. Unknown keys are ignored (`extra="ignore"`). Get config via
  `get_settings()` — don't read `os.environ` directly. `ExtractionSettings` and `NERSettings`
  are also independently instantiable (used directly in `entity_extraction.py`).
- **Postgres connections:** build the libpq string via `settings.database.conninfo` (a
  property on `DatabaseSettings`) and pass it to `psycopg.connect(...)`. Don't hand-assemble
  `host=... port=...` strings at call sites — `pgvector.py`, `init_db.py`, and `api/app.py`
  all go through `conninfo`.
- **Graph tuning:** `GraphSettings` (`GRAPH__*`) holds `db_path`,
  `merge_similarity_threshold` (0.90 — the cosine cutoff for `merge-graph` candidates) and
  `min_relation_weight` (0.50 — relations at/below this are treated as low-confidence). Route
  graph behavior through these rather than hard-coding constants.
- **Logging:** obtain loggers with `setup_logger(__name__)` from `constants.logger`. Logs
  go to `logs/app.log` (rotating, 5 MB × 3) plus stdout. Don't use bare `print` for
  diagnostics in library code (the CLI uses `typer.echo` for user-facing output).
- **Errors:** raise the typed exceptions in `constants/exceptions.py` (e.g.
  `ConfigurationError`, `TextExtractionError`, `GraphError`) rather than bare `Exception`.
  Every project exception now derives from a single `BaseError` root (including the
  ingestion/embedding/database/graph errors), so a broad `except BaseError` catches the whole
  family while specific subclasses stay targetable. Keep new exceptions under `BaseError`.
- **Imports:** keep the top-level style (`from graph.merge import ...`), consistent with the
  rest of the package and the editable install (`src/` is the package root).
- Heavy/optional imports inside CLI commands are imported lazily (inside the function) on
  purpose — keep that pattern so `--help` stays fast and optional paths don't force imports.

## Security notes

- Never commit a populated `.env` (it is gitignored). Secrets belong in `.env` only.
- `src/api/app.py` is a **local-first FastAPI app with no authentication**. It exposes
  `POST /ingest` (fire-and-forget background task; poll `GET /ingest/{task_id}`),
  `POST /query`, and `GET /health`. Ingestion runs arbitrary file paths from the request and
  there is no auth layer — do not expose it to a public network without adding authentication
  and input validation first. Flag this if asked to deploy it.

## Verification

There is **no test suite or linter configured** yet. After changes:
- Confirm the module imports cleanly from `src/` (e.g. `cd src && uv run python -c "import pipeline"`).
  Note that several modules (`api/app.py`, `retrieval/pgvector.py`, `llm/generator.py`,
  `config/init_db.py`) call `get_settings()` at **import time**, so importing them requires a
  populated `.env` (at minimum the `DATABASE__*` and `GENERATOR__*` required fields) — otherwise
  the import fails with a Pydantic `ValidationError`, not a code error.
- For ingestion/retrieval changes, a manual `ingest` then `ask` against a sample file in
  `data/` is the realistic smoke test (requires Postgres + Ollama running).
- If you add tests, place them under a top-level `tests/` and prefer `pytest`.

## Related projects (same workspace)

This repo is the consumer end of a pipeline:
`hgr-dataset` (builds the triplet dataset) → `finetuning_qwen3.5` (fine-tunes Qwen3-0.6B →
GGUF) → **Hybrid-RAG** (serves the model as the `hgr-triplet` Ollama model for extraction).
`KG_triplet_evaluation` scores that model's triplet output.
