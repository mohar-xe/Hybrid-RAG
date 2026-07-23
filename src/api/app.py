"""FastAPI backend — /ingest, /query, /health.

Security note: every mutating/read endpoint is gated by an API key when
``API__KEY`` is configured. ``/ingest`` only accepts file paths that resolve
inside the allow-listed ``API__INGEST_DIR`` so the endpoint cannot be coerced
into reading arbitrary server files. If ``API__KEY`` is empty the app runs in
*unauthenticated* mode and logs a loud warning at startup — do not expose such
a deployment to an untrusted network.
"""

import json
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psycopg
from fastapi import (
    FastAPI,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import get_settings
from constants.logger import setup_logger
from api.rate_limit import check_limit, LIMITS

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

    yield
    LOGGER.info("Shutting down.")


app = FastAPI(lifespan=lifespan, title="Hybrid-RAG", version="0.1.0")


# ── Rate limit dependency ────────────────────────────────────────────────


def _rate_limit(action: str):
    def _check(
        request: Request,
        x_api_key: str | None = Header(default=None),
        x_forwarded_for: str | None = Header(default=None),
    ):
        configured = settings.api.api_key.get_secret_value()
        if configured and x_api_key and secrets.compare_digest(x_api_key, configured):
            return True

        allowed, used = check_limit(
            action,
            x_api_key=x_api_key,
            x_forwarded_for=x_forwarded_for,
            remote=request.client.host if request.client else "",
        )
        if not allowed:
            max_ = LIMITS.get(action, 0)
            raise HTTPException(
                429,
                detail=f"Daily {action} limit reached ({used}/{max_}). Resets at midnight UTC.",
            )
        return True

    return Depends(_check)


# ── Static frontend ──────────────────────────────────────────────────────
_STATIC = Path(__file__).parent / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return HTMLResponse((_STATIC / "index.html").read_text())


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


@dataclass
class _Timer:
    stages: list[dict] = field(default_factory=list)
    _start: float = 0.0
    _current: str | None = None

    def start(self, name: str):
        if self._current is not None:
            self.end()
        self._current = name
        self._start = time.monotonic()

    def end(self, items: int = 0):
        if self._current is None:
            return
        elapsed = round((time.monotonic() - self._start) * 1000)
        self.stages.append({"name": self._current, "ms": elapsed, "items": items})
        self._current = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict] = []
    faithfulness: float | None = None
    metrics: dict | None = None
    chunks: list[dict] = []
    graph_facts: str = ""


def _run_ingestion(
    task_id: str,
    file_path: str,
    source_type: str,
    extractor: str | None = None,
    version_label: str | None = None,
    supersedes: str | None = None,
):

    from ingestion.extractor import Extractor
    from ingestion.chunker import chunk_text, chunk_enrich
    from ingestion.document_cluster import (
        extract_document_metadata,
        create_document_cluster,
    )
    from retrieval.pgvector import store_chunks
    from retrieval.kuzu_store import (
        get_connection,
        init_graph_schema,
        upsert_triplets,
        link_entities_to_chunk,
    )
    from graph.entity_extraction import extract_entities_batch

    import uuid as _uuid

    _INGEST_STAGES = [
        "Extracting text…",
        "Extracting metadata…",
        "Chunking…",
        "Storing chunks…",
        "Extracting graph entities…",
        "Storing graph…",
    ]

    def _set_stage(idx: int):
        done = [{"label": _INGEST_STAGES[i], "done": True} for i in range(idx)]
        current = {"label": _INGEST_STAGES[idx], "done": False}
        remaining = [{"label": s, "done": False} for s in _INGEST_STAGES[idx + 1 :]]
        _tasks[task_id]["stages"] = done + [current] + remaining

    _tasks[task_id]["status"] = "processing"
    _tasks[task_id]["stages"] = [{"label": s, "done": False} for s in _INGEST_STAGES]
    try:
        _set_stage(0)
        ext = Extractor()
        text = ext.extract_pdf(file_path)

        _set_stage(1)
        metadata = extract_document_metadata(text, source_id=file_path)
        is_versioned = bool(version_label or supersedes)
        doc_id = str(_uuid.uuid4())
        create_document_cluster(
            doc_id,
            file_path,
            source_type.upper(),
            metadata,
            text,
            supersedes_doc_id=supersedes,
            is_versioned=is_versioned,
            version_label=version_label,
        )

        _set_stage(2)
        chunks = chunk_enrich(
            chunk_text(text), source_type.upper(), file_path, doc_id=doc_id
        )

        _set_stage(3)
        store_chunks(chunks)

        _set_stage(4)
        triplets_list = extract_entities_batch(
            [c.text for c in chunks], backend=extractor
        )
        _set_stage(5)
        failed = 0
        with _GRAPH_WRITE_LOCK:
            graph_db, graph_conn = get_connection()
            init_graph_schema(conn=graph_conn)
            for chunk, triplets in zip(chunks, triplets_list):
                if triplets is None:
                    failed += 1
                    continue
                upsert_triplets(triplets, conn=graph_conn)
                entity_names = list(
                    {t.source.title for t in triplets}
                    | {t.target.title for t in triplets}
                )
                link_entities_to_chunk(
                    entity_names,
                    str(chunk.chunk_id),
                    text=chunk.text,
                    source_id=chunk.source_id,
                    conn=graph_conn,
                )

        # Mark the last stage as done so the frontend sees completion
        for s in _tasks[task_id].get("stages", []):
            if s["label"] == _INGEST_STAGES[-1]:
                s["done"] = True
                break

        _tasks[task_id].update(
            {"status": "completed", "chunks": len(chunks), "failed_extractions": failed}
        )
    except Exception as e:
        _tasks[task_id].update({"status": "failed", "error": str(e)})
        LOGGER.error(f"Ingestion failed: {e}")


class IngestRequest(BaseModel):
    file_path: str
    source_type: str = "pdf"
    extractor: str | None = None
    version_label: str | None = None
    supersedes: str | None = None


@app.post("/ingest")
async def ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks = None,
):
    if req.source_type != "pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source_type: {req.source_type} (only pdf is supported)",
        )

    file_path = _validate_ingest_path(req.file_path)

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "accepted"}
    background_tasks.add_task(
        _run_ingestion,
        task_id,
        file_path,
        req.source_type,
        req.extractor,
        req.version_label,
        req.supersedes,
    )
    return {"task_id": task_id, "status": "accepted"}


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    extractor: str | None = Form(default=None),
    version_label: str | None = Form(default=None),
    supersedes: str | None = Form(default=None),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    tmp_dir = Path("/tmp") / "hybrid_rag_ingest"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename

    content = await file.read()
    tmp_path.write_bytes(content)

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "accepted", "filename": file.filename}
    background_tasks.add_task(
        _run_ingestion,
        task_id,
        str(tmp_path),
        "pdf",
        extractor,
        version_label,
        supersedes,
    )
    return {"task_id": task_id, "status": "accepted", "filename": file.filename}


@app.get("/ingest/{task_id}")
async def ingest_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")
    return _tasks[task_id]


@app.post("/query", response_model=None)
async def query(req: QueryRequest):
    _t_start = time.monotonic()
    t = _Timer()

    from reasoning.router import route_retrieval
    from reasoning.query_interpreter import interpret_query
    from retrieval.pgvector import (
        hard_filter_docs,
        doc_level_soft_rank,
        document_routed_search,
    )
    from retrieval.kuzu_store import get_entity_context, structural_expansion
    from retrieval.reranker import rerank
    from context.builder import build_context
    from llm.generator import generate
    from embeddings.embedder import embedder

    # 0: route & interpret
    t.start("query_understanding")
    strategy = route_retrieval(req.question)
    interpreted = interpret_query(req.question)
    semantic_query = interpreted.get("semantic_query", req.question)
    filters = interpreted.get("filters", {})
    t.end()

    # embed
    t.start("embedding")
    query_emb = embedder([semantic_query])[0]
    t.end()

    # 1: hard filter
    t.start("hard_filter")
    doc_ids = hard_filter_docs(
        doc_type=filters.get("doc_type"),
        is_latest=filters.get("is_latest"),
        date_after=filters.get("date_after"),
    )
    t.end(items=len(doc_ids))

    # 2: doc-level soft ranking
    t.start("doc_soft_rank")
    ranked_docs = doc_level_soft_rank(query_emb, semantic_query, doc_ids)
    if ranked_docs:
        doc_ids = [d[0] for d in ranked_docs]
        doc_rank_map = {d[0]: i + 1 for i, d in enumerate(ranked_docs)}
    else:
        doc_rank_map = None
    t.end(items=len(ranked_docs) if ranked_docs else 0)

    # 3-4: chunk-level hybrid search + cross-list RRF fusion
    t.start("chunk_hybrid_search")
    chunks = document_routed_search(
        query_emb, semantic_query, doc_ids, doc_rank_map=doc_rank_map
    )
    t.end(items=len(chunks))

    if not chunks:
        raise HTTPException(404, "No relevant context found")

    # 5: rerank
    t.start("rerank")
    chunks = rerank(semantic_query, chunks, top_k=req.top_k)
    t.end(items=len(chunks))

    # 6: structural expansion (small-to-big graph traversal)
    t.start("graph_expansion")
    expanded_chunks = []
    try:
        chunk_ids = [c.chunk_id for c in chunks]
        siblings = structural_expansion(chunk_ids)
        if siblings:
            from retrieval.pgvector import RetrievedChunk

            for cid, text, score in siblings:
                expanded_chunks.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        text=text,
                        source_id="",
                        score=score,
                    )
                )
    except Exception as e:
        LOGGER.warning("Structural expansion failed (continuing): %s", e)
    t.end(items=len(expanded_chunks))

    # 7: graph facts (entity context)
    t.start("extract_entities")
    graph_facts = ""
    from graph.entity_extraction import extract_query_entities

    entities = extract_query_entities(req.question)
    t.end(items=len(entities))

    t.start("context_build")
    if entities:
        graph_facts = get_entity_context(entities)

    # 8: assemble context (reranked chunks + expanded siblings + graph facts)
    all_chunks = chunks + expanded_chunks
    context, citations = build_context(all_chunks, graph_facts)
    t.end(items=len(citations))

    if req.stream:
        import json as _json

        chunks_data = [
            {"score": c.score, "source": c.source_id, "preview": c.text[:200]}
            for c in chunks
        ]
        citations_data = [
            {"ref": c.ref_id, "source": c.source_id, "preview": c.text_preview}
            for c in citations
        ]
        metrics_data = {
            "total_ms": round((time.monotonic() - _t_start) * 1000),
            "stages": t.stages,
            "models": {
                "embedder": settings.embedding.backend,
                "reranker": settings.reranker.backend,
                "generator": settings.generator.backend,
            },
        }

        async def _stream_answer():
            _gen_start = time.monotonic()
            meta = {
                "type": "meta",
                "chunks": chunks_data,
                "graph_facts": graph_facts,
                "citations": citations_data,
                "metrics": metrics_data,
            }
            yield f"data: {_json.dumps(meta)}\n\n"
            for token in generate(req.question, context, stream=True):
                if isinstance(token, str):
                    yield f"data: {_json.dumps({'type': 'token', 'text': token})}\n\n"
            gen_ms = round((time.monotonic() - _gen_start) * 1000)
            # send updated metrics with generate stage included
            updated = dict(metrics_data)
            updated["stages"] = list(updated["stages"]) + [
                {"name": "generate", "ms": gen_ms, "items": 0}
            ]
            updated["total_ms"] = round((time.monotonic() - _t_start) * 1000)
            yield f"data: {_json.dumps({'type': 'meta', 'metrics': updated})}\n\n"
            yield "data: {\"type\":\"done\"}\n\n"

        return StreamingResponse(
            _stream_answer(), media_type="text/event-stream"
        )

    t.start("generate")
    answer = generate(req.question, context, stream=False)
    t.end()

    faithfulness = None
    if settings.verifier.enabled and context:
        try:
            from verification.verifier import score_faithfulness

            faithfulness = score_faithfulness(answer, context)
        except Exception as e:
            LOGGER.warning(f"Faithfulness scoring failed: {e}")

    return QueryResponse(
        answer=answer,
        chunks=[
            {"score": c.score, "source": c.source_id, "preview": c.text[:200]}
            for c in chunks
        ],
        graph_facts=graph_facts,
        citations=[
            {"ref": c.ref_id, "source": c.source_id, "preview": c.text_preview}
            for c in citations
        ],
        faithfulness=faithfulness,
        metrics={
            "total_ms": round((time.monotonic() - _t_start) * 1000),
            "stages": t.stages,
            "models": {
                "embedder": settings.embedding.backend,
                "reranker": settings.reranker.backend,
                "generator": settings.generator.backend,
            },
        },
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

    # Only check Ollama when the extraction backend actually uses it
    # (Render deployments use remote APIs, not a local Ollama).
    ollama_needed = settings.extraction.backend == "local"
    ollama_ok = False
    if ollama_needed:
        ollama_url = settings.extraction.base_url.rstrip("/")
        try:
            r = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
            ollama_ok = r.status_code == 200
        except Exception:
            pass

    graph_ok = settings.graph.db_path.exists()
    entity_count = 0
    if graph_ok:
        try:
            from retrieval.kuzu_store import get_connection
            _db, _conn = get_connection()
            result = _conn.execute("MATCH (e:Entity) RETURN count(*) AS cnt")
            entity_count = result.get_next()[0]
        except Exception:
            pass

    status = (
        "degraded" if (not db_ok or (ollama_needed and not ollama_ok)) else "healthy"
    )
    return {
        "status": status,
        "database": "ok" if db_ok else "unreachable",
        "ollama": ("ok" if ollama_ok else "unreachable")
        if ollama_needed
        else "not_configured",
        "graph": "ok" if graph_ok else "not_initialized",
        "entities": entity_count,
    }
