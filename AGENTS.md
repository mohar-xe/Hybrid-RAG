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

- **Implemented:** ingestion (PDF/YouTube/audio extraction → normalize → **recursive
  token-approx chunking** → keyphrase → embed), pgvector storage + schema init,
  config/logging/exception infrastructure, Ollama embeddings, LLM entity/triplet extraction
  (local + remote backends), Kùzu graph store, near-duplicate node merge, **coarse→fine
  cluster-routed dense retrieval**, K-Means/medoid clustering via an explicit `reindex` command,
  **cross-encoder reranking**, Kùzu graph-context lookup, heuristic query router, context
  assembly + citations, OpenAI-compatible generation, a FastAPI app (`/ingest` async +
  status, `/query`, `/health`), and a **Gradio demo UI** (`src/demo/app.py`) that exposes
  retrieval internals for technical demos.
- **Planned / scaffolded (config may exist but logic is not wired yet):** NLI self-verification
  (`VerifierSettings` exists; `/query` returns `faithfulness: None`), retrieval quality scoring,
  HyDE, RAPTOR summaries, graph community summaries, streaming `/query` responses (the `stream`
  flag is accepted but ignored), real-time graph visualization, and the evaluation suite.
- **Removed (do not reintroduce without discussion):** the old semantic/hierarchical chunker
  (paragraph-embed → adjacency-cluster → spaCy entity-aware split) and flat vector/BM25
  retrieval — both replaced by the staged pipeline below.

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
uv run python pipeline.py reindex                # cluster chunks + mark medoids (run after ingest)
uv run python pipeline.py ask "<question>" --verbose
uv run python pipeline.py merge-graph            # dry run: lists candidates
uv run python pipeline.py merge-graph --apply    # actually merges
uv run python pipeline.py merge-graph --threshold 0.95   # override similarity cutoff
```

Typical end-to-end order: **`ingest` → `reindex` → `ask`** (and `merge-graph --apply` any time
after ingest to deduplicate graph entities). `reindex` is an explicit indexing step like
`merge-graph`: it (re)computes `cluster_id`/`is_medoid` over all stored chunks. Until it runs,
retrieval still works — it falls back to flat global ANN (see "Retrieval pipeline").

`ingest` has **two independent phases**, separately toggleable:
- **store** (`--store`/`--no-store`): extract → chunk → embed → pgvector. This is the slow part
  (CPU embedding).
- **graph** (`--graph`/`--no-graph`): triplet extraction → Kùzu.

Because embedding is expensive, a transient extraction failure (e.g. a bad/empty DeepSeek
response) should **not** force a re-embed. Re-run just the graph phase against already-stored
chunks: `ingest <path> --no-store --graph` (it loads chunks via
`pgvector.get_chunks_by_source`). The graph loop is also **per-chunk resilient** — one failed
extraction is logged and skipped, not fatal, and `_parse_json` treats an empty response as zero
triplets.

- The **root `main.py` is a placeholder stub** — the real entry point is `src/pipeline.py`
  (a Typer app with `ingest`, `ask`, `reindex`, `merge-graph` commands).
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
retrieval pipeline (coarse→fine routing, pre/post-rerank candidates, final chunks, graph facts)
with toggleable features (reranking, knowledge graph). Use it to demonstrate system design
decisions to technical audiences.

## Architecture / layout (`src/`)

```
config/      Pydantic settings (settings.py) + DB schema init/migrations (init_db.py)
constants/   logger.py (rotating file + console) + exceptions.py (typed hierarchy)
ingestion/   extractor, normalize, chunker (recursive splitter), chunk_schema  [implemented]
embeddings/  embedder.py — nomic-embed-text via Ollama, 256-dim (Matryoshka)
graph/       entity_extraction.py, schema.py, merge.py    [implemented]
retrieval/   pgvector.py (cluster-routed dense search), cluster.py (K-Means/medoids),
             reranker.py (cross-encoder), kuzu_store.py   [implemented]
context/     builder.py — context assembly + citations    [implemented]
llm/         generator.py — OpenAI-compatible generation   [implemented]
reasoning/   router.py — heuristic query routing [implemented]; verifier [planned]
api/         app.py — FastAPI surface (/ingest, /query, /health) [implemented core]
demo/        app.py — Gradio UI for interactive demos with retrieval internals [implemented]
pipeline.py  Typer CLI (ingest, ask, reindex, merge-graph) — real entry point
```

## Chunking & embedding (ingestion)

`ingestion/chunker.py` is a **RecursiveCharacterTextSplitter** (`chunk_text`): it splits on a
separator priority list `["\n\n", "\n", ". ", " "]`, descending to the next separator for any
piece still over budget and hard-splitting on characters as a last resort (`_atomic_splits`),
then greedily re-packs pieces with a sliding-window overlap (`_merge_splits`). Sizes are
**character-approximated tokens** (`CHARS_PER_TOKEN = 4`): `CHUNK_CHARS = 1200` (~300 tokens),
`OVERLAP_CHARS = 200` (~50 tokens). This bounds every chunk *before* embedding, which is the
whole point — the previous paragraph-embed-then-cluster design could feed 25k-token blobs to
Ollama and 400. There is **no tokenizer dependency** and **no spaCy/clustering** in chunking
anymore.

`embeddings/embedder.py` (`nomic-embed-text`, 2048-token context):
- Each input is capped to `MAX_INPUT_CHARS = 8000` client-side and `truncate=True` is passed to
  Ollama — a defence-in-depth against over-long inputs (a single over-long input, or a *batch*
  of them, otherwise 400s).
- `BATCH_SIZE = 16`. Ollama CPU embedding shows **no batch speedup** (~constant s/input) and a
  very large batch becomes one long request that monopolizes the runner and looks hung. Don't
  raise this without measuring.
- Output is truncated to 256 dims (Matryoshka) and the chunker L2-normalizes stored vectors.
- `normalize.remove_control_chars` (and a defensive strip in `chunk_text`) removes NUL/control
  bytes — PostgreSQL `text` rejects NUL, which PDF extraction frequently injects.

## Retrieval pipeline (coarse→fine + rerank)

Flat vector/BM25 retrieval is **gone**. `ask`/`/query` now run:

1. Embed the query (`embedder`).
2. `retrieval.pgvector.cluster_routed_search(emb)`:
   - **Coarse**: ANN over medoids only (`WHERE is_medoid`) → top 10.
   - **Gate**: if best medoid similarity < `0.35` *or there are no medoids yet* (i.e. `reindex`
     hasn't run), fall back to flat global ANN.
   - **Select** the top 5 distinct clusters by medoid similarity.
   - **Fine**: cluster-filtered ANN (`cluster_id = ANY(top5)`) → top 15.
   - **Global fallback**: a parallel unfiltered ANN → top 5, merged + de-duplicated with the
     fine results (keep higher score). Yields up to ~20 candidates.
3. `retrieval.reranker.rerank(query, candidates)` (cross-encoder
   `cross-encoder/ms-marco-MiniLM-L-6-v2`, lazy-loaded) → final top-k (`RERANKER__TOP_K`, default
   5). **Rerank always runs now** (Stage 4 is unconditional), not gated on the router.
4. Optional Kùzu graph facts (gated on the router's `use_graph`), then context build + generate.

Clustering (`retrieval/cluster.py`, invoked by `reindex`): spherical K-Means over the
L2-normalized 256-d chunk embeddings with `K = max(3, round(sqrt(n)))`; the **medoid** of each
cluster is the actual chunk with max cosine to the centroid. `cluster_id`/`is_medoid` are
persisted on the `chunks` table (`init_db` adds them via idempotent `ALTER ... ADD COLUMN IF NOT
EXISTS`, plus a partial medoid index and a `cluster_id` index). `reasoning/router.py` still
classifies complexity and provides `use_graph`, but its `use_bm25`/`top_k`/`use_reranker` fields
no longer drive retrieval.

## Graph store (Kùzu)

`retrieval/kuzu_store.py` defines four tables: `Entity` (name PK, entity_type), `Chunk`
(chunk_id PK, text, source_id), `RELATES_TO` (Entity→Entity, relation_type, weight), and
`MENTIONED_IN` (Entity→Chunk). Notes for anyone touching this:
- `upsert_triplets` MERGEs entities and their `RELATES_TO` edge.
- `link_entities_to_chunk(entity_names, chunk_id, text, source_id)` **MERGEs the `Chunk` node
  itself** before creating `MENTIONED_IN` edges — nothing else inserts `Chunk` nodes, so if you
  drop that MERGE the edges silently never form (the entity→chunk `MATCH` finds no chunk). Pass
  `text`/`source_id` so the node is populated on creation.
- `get_entity_context` does a **multi-hop** BFS over `RELATES_TO`: it follows up to
  `GRAPH__MAX_HOPS` (default 2) edges out from each seed entity, **bidirectionally**, keeping
  only relations with weight `> GRAPH__MIN_RELATION_WEIGHT`. This is what lets the graph answer
  "bridge" questions — seed `A` → `A rel B` (hop 1) → `B rel C` (hop 2) — that a single hop can
  never reach. Fan-out is capped per node per hop (`GRAPH__PER_HOP_NEIGHBORS`) and the total
  facts are capped (`GRAPH__MAX_FACTS`); facts come back closest-hop-first, de-duplicated.
  `hops=1` reproduces the old single-hop behavior (now also bidirectional). `_one_hop_neighbors`
  is the private per-hop expansion helper.
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
prompt (`_build_system_prompt`), JSON parsing (`_parse_json` — returns `[]` for an empty
response instead of raising), and an **idempotent schema check** (`validate_triplets` —
validates raw dicts against the `Triplet` schema, drops invalid ones, and passes through items
that are already validated `Triplet` instances). Keep that shared path intact, and register any
third backend in `_BACKENDS`.

Concurrency & thinking:
- `extract_entities_batch(texts, backend, max_workers)` runs extraction **concurrently**
  (`ThreadPoolExecutor`, I/O-bound calls), order-preserved, returning `None` for any text that
  errored (the caller counts/skips it). Concurrency defaults to `NER__CONCURRENCY` /
  `EXTRACTION__CONCURRENCY` (both 10). The ingest graph phase calls this in batches and writes
  results to Kùzu **serially** (a Kùzu connection is not thread-safe).
- `NER__DISABLE_THINKING` (default `true`) sends `{"thinking": {"type": "disabled"}}` to the
  DeepSeek API. DeepSeek's hybrid models default to reasoning, which burns the token budget and
  often leaves `content` empty — with thinking on, extraction yielded ~2 triplets across 145
  chunks; with it off, ~1000. Leave this on unless a backend rejects the field.

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
  `merge_similarity_threshold` (0.90 — the cosine cutoff for `merge-graph` candidates),
  `min_relation_weight` (0.50 — relations at/below this are treated as low-confidence), and the
  multi-hop retrieval knobs `max_hops` (2), `max_facts` (30 — total fact cap), and
  `per_hop_neighbors` (10 — per-node fan-out per hop). Route graph behavior through these rather
  than hard-coding constants.
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
