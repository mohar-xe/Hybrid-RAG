"""FastAPI backend — /ingest, /query, /health.

Security note: every mutating/read endpoint is gated by an API key when
``API__KEY`` is configured. ``/ingest`` only accepts file paths that resolve
inside the allow-listed ``API__INGEST_DIR`` so the endpoint cannot be coerced
into reading arbitrary server files. If ``API__KEY`` is empty the app runs in
*unauthenticated* mode and logs a loud warning at startup — do not expose such
a deployment to an untrusted network.
"""

import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()

_tasks: dict[str, dict] = {}
# Kùzu allows a single writer; background ingest tasks run in a threadpool, so
# serialize all graph writes within this process. (Across multiple worker
# processes an external queue is still required — see README.)
_GRAPH_WRITE_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the Postgres schema (extension, chunks table, indexes) so a
    # fresh API-only deployment can ingest without first running the CLI.
    from config.init_db import init_db

    try:
        init_db()
        LOGGER.info("Database schema initialized.")
    except Exception as e:
        LOGGER.warning(f"Database init failed (continuing): {e}")

    if not settings.api.api_key.get_secret_value():
        LOGGER.warning(
            "API__KEY is not set — the API is running UNAUTHENTICATED. Set "
            "API__KEY in .env before exposing this service."
        )
    yield
    LOGGER.info("Shutting down.")


app = FastAPI(lifespan=lifespan, title="Hybrid-RAG", version="0.1.0")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Gate an endpoint behind the configured API key.

    No-op when ``API__KEY`` is empty (local/unauthenticated mode). Uses a
    constant-time comparison to avoid leaking the key via timing.
    """
    configured = settings.api.api_key.get_secret_value()
    if not configured:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _validate_ingest_path(file_path: str) -> str:
    """Resolve ``file_path`` and ensure it stays within ``API__INGEST_DIR``.

    Prevents path-traversal / arbitrary file reads: a request may only ingest
    files located inside the allow-listed directory. Returns the resolved
    absolute path.
    """
    base = settings.api.ingest_dir.resolve()
    candidate = Path(file_path)
    candidate = candidate if candidate.is_absolute() else base / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise HTTPException(
            status_code=400,
            detail=f"file_path must resolve inside the allowed ingest directory ({base}).",
        )
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return str(resolved)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    faithfulness: float | None = None


def _run_ingestion(task_id: str, file_path: str, source_type: str, extractor: str | None = None):
    from ingestion.extractor import Extractor
    from ingestion.chunker import chunk_text, chunk_enrich
    from retrieval.pgvector import store_chunks
    from retrieval.kuzu_store import get_connection, init_graph_schema, upsert_triplets, link_entities_to_chunk
    from graph.entity_extraction import extract_entities_batch

    _tasks[task_id]["status"] = "processing"
    try:
        ext = Extractor()
        if source_type == "pdf":
            text = ext.extract_pdf(file_path)
        elif source_type == "youtube":
            text = ext.yt_subtitle_extraction(file_path)
        else:
            text = ext.reel_subtitle_extraction(file_path)

        chunks = chunk_enrich(chunk_text(text), source_type.upper(), file_path)
        store_chunks(chunks)

        # Extract triplets concurrently (I/O-bound), then write to Kùzu serially
        # under the process-wide lock (the connection is not thread-safe and
        # Kùzu permits a single writer).
        triplets_list = extract_entities_batch([c.text for c in chunks], backend=extractor)
        failed = 0
        with _GRAPH_WRITE_LOCK:
            graph_db, graph_conn = get_connection()
            init_graph_schema(conn=graph_conn)
            for chunk, triplets in zip(chunks, triplets_list):
                if triplets is None:  # extraction errored for this chunk
                    failed += 1
                    continue
                upsert_triplets(triplets, conn=graph_conn)
                entity_names = list({t.source.title for t in triplets} | {t.target.title for t in triplets})
                link_entities_to_chunk(
                    entity_names, str(chunk.chunk_id), text=chunk.text,
                    source_id=chunk.source_id, conn=graph_conn,
                )

        _tasks[task_id].update({"status": "completed", "chunks": len(chunks), "failed_extractions": failed})
    except Exception as e:
        _tasks[task_id].update({"status": "failed", "error": str(e)})
        LOGGER.error(f"Ingestion failed: {e}")


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest(
    file_path: str,
    source_type: str = "pdf",
    extractor: str | None = None,
    background_tasks: BackgroundTasks = None,
):
    if source_type not in ("pdf", "youtube", "audio"):
        raise HTTPException(status_code=400, detail=f"Unknown source_type: {source_type}")

    # File-based sources must live inside the allow-listed directory. YouTube
    # takes a video id (not a path), so it is exempt from path validation.
    if source_type in ("pdf", "audio"):
        file_path = _validate_ingest_path(file_path)

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "accepted"}
    background_tasks.add_task(_run_ingestion, task_id, file_path, source_type, extractor)
    return {"task_id": task_id, "status": "accepted"}


@app.get("/ingest/{task_id}", dependencies=[Depends(require_api_key)])
async def ingest_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")
    return _tasks[task_id]


@app.post("/query", response_model=None, dependencies=[Depends(require_api_key)])
async def query(req: QueryRequest):
    from reasoning.router import route_retrieval
    from retrieval.pgvector import cluster_routed_search
    from retrieval.kuzu_store import get_entity_context
    from retrieval.reranker import rerank
    from context.builder import build_context
    from llm.generator import generate
    from embeddings.embedder import embedder

    strategy = route_retrieval(req.question)

    emb = embedder([req.question])[0]
    chunks = cluster_routed_search(emb)

    if not chunks:
        raise HTTPException(404, "No relevant context found")

    # Stage 4: rerank routed candidates to the requested top-k.
    chunks = rerank(req.question, chunks, top_k=req.top_k)

    graph_facts = ""
    if strategy["use_graph"]:
        entities = [w for w in req.question.split() if w[0].isupper() and len(w) > 1]
        if entities:
            graph_facts = get_entity_context(entities)

    context, citations = build_context(chunks, graph_facts)

    # Streaming: emit the answer tokens as they arrive (citations/faithfulness
    # are omitted from the stream — clients needing them should set stream=false).
    if req.stream:
        return StreamingResponse(
            generate(req.question, context, stream=True), media_type="text/plain"
        )

    answer = generate(req.question, context)

    # Optional gated faithfulness check (VERIFIER__ENABLED).
    faithfulness = None
    if settings.verifier.enabled and context:
        try:
            from verification.verifier import score_faithfulness

            faithfulness = score_faithfulness(answer, context)
        except Exception as e:  # never let verification break a response
            LOGGER.warning(f"Faithfulness scoring failed: {e}")

    return QueryResponse(
        answer=answer,
        citations=[{"ref": c.ref_id, "source": c.source_id, "preview": c.text_preview} for c in citations],
        faithfulness=faithfulness,
    )


@app.get("/health")
async def health():
    db_ok = False
    try:
        with psycopg.connect(settings.database.conninfo) as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    # Use the configured Ollama endpoint rather than a hardcoded localhost URL.
    ollama_ok = False
    ollama_url = settings.extraction.base_url.rstrip("/")
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    graph_ok = settings.graph.db_path.exists()

    status = "healthy" if (db_ok and ollama_ok) else "degraded"
    return {
        "status": status,
        "database": "ok" if db_ok else "unreachable",
        "ollama": "ok" if ollama_ok else "unreachable",
        "graph": "ok" if graph_ok else "not_initialized",
    }
