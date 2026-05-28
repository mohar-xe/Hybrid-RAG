# Hybrid-RAG

A hybrid retrieval-augmented generation system that combines vector search with knowledge graph traversal for question answering.

Python 3.12+ · uv · PostgreSQL/pgvector · KùzuDB

---

## What Exists

### Ingestion Pipeline (implemented)

- **Text extraction** — PDF (pymupdf), YouTube subtitles (youtube-transcript-api), audio/video transcription (pywhispercpp/Whisper)
- **Text normalization** — 12-stage cleaning pipeline: unicode normalization, hyphenation repair, broken line joining, page number removal, header/footer detection, whitespace collapse
- **Hierarchical chunking** — paragraphs grouped by embedding cosine similarity, oversized groups split at sentence boundaries using spaCy NER to avoid cutting through entities
- **Keyphrase extraction** — YAKE unsupervised keyphrases per chunk
- **Embeddings** — nomic-embed-text via Ollama, batched, truncated to 256 dimensions (Matryoshka)

### Storage Schema (implemented)

- **pgvector** — chunks table with HNSW index (cosine) + GIN index (tsvector for full-text search)
- **Schema init** — automated table creation, extension enabling, index building

### Infrastructure (implemented)

- **Configuration** — Pydantic BaseSettings with nested sections (database, graph, embedding, generator), loaded from `.env`
- **Logging** — rotating file handler (5MB, 3 backups) + console, consistent format with source location
- **Exceptions** — two-tier typed hierarchy (global errors + domain-specific extraction errors)

---

## What Will Be Implemented

### Retrieval

- Vector retrieval with hybrid scoring (cosine similarity + BM25 tsvector ranking)
- Knowledge graph storage and traversal (KùzuDB, Cypher queries, multi-hop entity lookup)
- Entity extraction using a locally fine-tuned FLM 450M model
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) to re-score top candidates
- Retrieval quality scoring — check if retrieved context is relevant before generating
- HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer, embed it, use that for retrieval
- Graph community detection and per-community summaries for broad/thematic queries
- RAPTOR-style hierarchical summaries (cluster chunks → summarize → re-cluster at multiple levels)
- Adaptive routing — classify query complexity, skip expensive retrieval paths for simple questions

### Generation

- LLM generation via any OpenAI-compatible endpoint (NIM, OpenRouter, Ollama)
- Citation/provenance — inline source references in generated answers
- NLI-based self-verification (DeBERTa entailment check on generated claims vs context)

### Pipeline & API

- CLI entry point: `ingest <file>` and `ask <question>`
- FastAPI backend with `/ingest`, `/query`, `/health` endpoints
- Real-time graph visualization

### Evaluation

- Retrieval metrics: Recall@K, MRR, NDCG
- Generation metrics: faithfulness, answer relevancy (DeepEval/RAGAS)
- Component ablation (measure contribution of each retrieval path)
- Benchmarks: HotpotQA, MuSiQue, Natural Questions

---

## Project Structure

```
src/
├── config/          # Settings (Pydantic) + DB schema init
├── constants/       # Logger + exception hierarchy
├── ingestion/       # Extractor, normalize, chunker, chunk schema
├── embeddings/      # nomic-embed-text via Ollama
├── graph/           # (planned) Entity extraction + community summaries
├── retrieval/       # (planned) pgvector, KùzuDB, hybrid, reranker
├── context/         # (planned) Context builder with citations
├── llm/             # (planned) Generator
├── reasoning/       # (planned) Router, verifier
└── pipeline.py      # (planned) CLI entry point
```

---

## Setup

```bash
git clone https://github.com/yourusername/Hybrid-RAG.git
cd Hybrid-RAG
uv sync
cp .env.example .env
# Fill in .env with database credentials and model settings
```

---

## Dependencies

- pymupdf — PDF extraction
- pywhispercpp — audio transcription
- youtube-transcript-api — YouTube subtitles
- ollama — embeddings (nomic-embed-text)
- psycopg — PostgreSQL/pgvector
- pydantic-settings — configuration
- spacy — sentence/entity boundary detection for chunking
- yake — keyphrase extraction

---

## License

MIT
