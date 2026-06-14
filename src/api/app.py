"""FastAPI backend — /query endpoint."""

import uuid
from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()

_tasks: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with psycopg.connect(settings.database.conninfo) as conn:
            conn.execute("SELECT 1")
        LOGGER.info("Database connection verified.")
    except Exception as e:
        LOGGER.warning(f"Database not reachable: {e}")
    yield
    LOGGER.info("Shutting down.")


app = FastAPI(lifespan=lifespan, title="Hybrid-RAG", version="0.1.0")


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
    from graph.entity_extraction import extract_entities

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

        # One Kùzu handle for the whole task; `graph_db` kept alive so the
        # Database isn't GC'd while `graph_conn` is in use.
        graph_db, graph_conn = get_connection()
        init_graph_schema(conn=graph_conn)
        for chunk in chunks:
            triplets = extract_entities(chunk.text, backend=extractor)
            upsert_triplets(triplets, conn=graph_conn)
            entity_names = list({t.source.title for t in triplets} | {t.target.title for t in triplets})
            link_entities_to_chunk(entity_names, str(chunk.chunk_id), text=chunk.text, source_id=chunk.source_id, conn=graph_conn)

        _tasks[task_id].update({"status": "completed", "chunks": len(chunks)})
    except Exception as e:
        _tasks[task_id].update({"status": "failed", "error": str(e)})
        LOGGER.error(f"Ingestion failed: {e}")


@app.post("/ingest")
async def ingest(
    file_path: str,
    source_type: str = "pdf",
    extractor: str | None = None,
    background_tasks: BackgroundTasks = None,
):
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "accepted"}
    background_tasks.add_task(_run_ingestion, task_id, file_path, source_type, extractor)
    return {"task_id": task_id, "status": "accepted"}


@app.get("/ingest/{task_id}")
async def ingest_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")
    return _tasks[task_id]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    from reasoning.router import route_retrieval
    from retrieval.pgvector import hybrid_search, vector_search
    from retrieval.kuzu_store import get_entity_context
    from context.builder import build_context
    from llm.generator import generate
    from embeddings.embedder import embedder

    strategy = route_retrieval(req.question)

    if strategy["use_bm25"]:
        chunks = hybrid_search(req.question, top_k=strategy["top_k"])
    else:
        emb = embedder([req.question])[0]
        chunks = vector_search(emb, top_k=strategy["top_k"])

    if not chunks:
        raise HTTPException(404, "No relevant context found")

    graph_facts = ""
    if strategy["use_graph"]:
        entities = [w for w in req.question.split() if w[0].isupper() and len(w) > 1]
        if entities:
            graph_facts = get_entity_context(entities)

    context, citations = build_context(chunks, graph_facts)
    answer = generate(req.question, context)

    return QueryResponse(
        answer=answer,
        citations=[{"ref": c.ref_id, "source": c.source_id, "preview": c.text_preview} for c in citations],
        faithfulness=None,
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

    ollama_ok = False
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
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