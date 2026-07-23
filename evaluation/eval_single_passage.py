#!/usr/bin/env python3
"""
Single-passage evaluation script for Hybrid-RAG.

Configuration (overridable via env vars / --options):
  - Generation:   DeepSeek API (user-provided key)
  - Embedding:    Mistral API (mistral-embed)
  - NER:          Mistral API (mistral-small-latest, entity extraction for graph mode)
  - Verifier:     Mistral API (mistral-small-latest, faithfulness scoring)
  - Reranker:     Mistral API (mistral-small-latest, LLM-as-reranker)

Takes 1 HotpotQA query from the cached n=1 selection, ingests its passage
corpus into pgvector, runs retrieval + generation across all configurations
(semantic / semantic_bm25 / all_three × rerank on/off, plus closed-book),
and reports ALL metrics: F1, EM, token recall, hit@k, answer-in-context,
faithfulness (verifier), and latency.

Usage:
    uv run python evaluation/eval_single_passage.py
    uv run python evaluation/eval_single_passage.py --with-graph   # include all_three
    uv run python evaluation/eval_single_passage.py --data-file evaluation/data/selected_distractor_validation_n10_seed42.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ===========================================================================
# Phase 0 — Bootstrap: paths + env vars BEFORE any project imports
# ===========================================================================

_HERE = Path(__file__).resolve().parent             # evaluation/
_ROOT = _HERE.parent                                 # project root
_SRC = _ROOT / "src"                                 # project source
# Insert src FIRST so ``config.settings`` resolves to ``src/config/settings.py``
# rather than the evaluation harness's ``evaluation/config.py``.
for _p in (_SRC, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)   # move to front in case it was already present
    sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Read the existing Mistral API key from .env so we can reuse it for NER,
# verifier, and reranker (all point at the same Mistral API).
# ---------------------------------------------------------------------------
_MISTRAL_KEY = ""
_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line.startswith("EMBEDDING__API_KEY="):
            _raw = _line.split("=", 1)[1].strip()
            _MISTRAL_KEY = _raw.strip("\"'").strip()
            break

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="Hybrid-RAG single-passage evaluation",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_parser.add_argument(
    "--with-graph", action="store_true",
    help="Include all_three (dense+sparse+graph) mode (requires Kùzu + NER endpoint)",
)
_parser.add_argument(
    "--top-k", type=int, default=5,
    help="Number of final chunks fed to the generator (default: 5)",
)
_parser.add_argument(
    "--candidate-k", type=int, default=20,
    help="Per-retriever candidate pool size (default: 20)",
)
_parser.add_argument(
    "--data-file", type=str, default=None,
    help="Path to a cached HotpotQA JSON selection (default: n=1, seed=42)",
)
_parser.add_argument(
    "--skip-ingest", action="store_true",
    help="Skip ingestion (use already-ingested corpus)",
)
_parser.add_argument(
    "--direct-only", action="store_true",
    help="Run only the closed-book (direct) baseline for comparison",
)
_parser.add_argument(
    "--clean", action="store_true", default=True,
    help="Truncate existing chunks/clusters before ingesting (default: true)",
)
_parser.add_argument(
    "--no-clean", action="store_false", dest="clean",
    help="Keep existing data in the DB",
)
_args = _parser.parse_args()

# ---------------------------------------------------------------------------
# Environment variable overrides (pydantic-settings reads env vars before .env)
# ---------------------------------------------------------------------------

# Generation: DeepSeek API
os.environ.setdefault("GENERATOR__BASE_URL", "https://api.deepseek.com/v1")
os.environ["GENERATOR__MODEL"] = "deepseek-chat"
os.environ["GENERATOR__API_KEY"] = "sk-608da4f8e4864b7d82400c688abef4af"
os.environ["GENERATOR__BACKEND"] = "api"
os.environ["GENERATOR__FALLBACK_ENABLED"] = "false"

# Embedding: Mistral API (already configured in .env, just ensuring overrides)
os.environ.setdefault("EMBEDDING__BACKEND", "api")
os.environ.setdefault("EMBEDDING__API_BASE_URL", "https://api.mistral.ai/v1")
os.environ.setdefault("EMBEDDING__API_MODEL", "mistral-embed")

# NER (entity extraction from text chunks for graph): Mistral API
os.environ["NER__BASE_URL"] = "https://api.mistral.ai/v1"
os.environ["NER__MODEL"] = "mistral-small-latest"
os.environ["NER__API_KEY"] = _MISTRAL_KEY
os.environ["NER__BATCH_SIZE"] = "1"
os.environ["EXTRACTION__BACKEND"] = "deepseek"

# Verifier (faithfulness scoring): Mistral API via generic chat backend
os.environ["VERIFIER__BACKEND"] = "api"
os.environ["VERIFIER__API_BASE_URL"] = "https://api.mistral.ai/v1"
os.environ["VERIFIER__API_MODEL"] = "mistral-small-latest"
os.environ["VERIFIER__API_KEY"] = _MISTRAL_KEY
os.environ["VERIFIER__ENABLED"] = "true"
os.environ["VERIFIER__FALLBACK_ENABLED"] = "false"

# Reranker: Mistral API (LLM-as-reranker via chat completions)
os.environ["RERANKER__BACKEND"] = "api"
os.environ["RERANKER__API_BASE_URL"] = "https://api.mistral.ai/v1"
os.environ["RERANKER__API_MODEL"] = "mistral-small-latest"
os.environ["RERANKER__API_KEY"] = _MISTRAL_KEY
os.environ["RERANKER__FALLBACK_ENABLED"] = "false"

# Graph DB path (anchored so every command opens the same file)
if _args.with_graph:
    os.environ.setdefault("GRAPH__DB_PATH", str(_SRC / "data" / "kuzu_db"))


# ===========================================================================
# Phase 1 — Imports
# ===========================================================================

from config.settings import get_settings
from config.init_db import init_db
from constants.logger import setup_logger
from embeddings.embedder import embedder
from retrieval.search import search
from context.builder import build_context
from llm.generator import generate
from evaluation.metrics import (
    f1_score,
    exact_match,
    token_recall,
    retrieval_hit_at_k,
    answer_in_context,
)
from evaluation.corpus import ingest_corpus

LOGGER = setup_logger("eval_single_passage")

# Disable noisy loggers during eval
import logging
logging.getLogger("retrieval.pgvector").setLevel(logging.WARNING)
logging.getLogger("retrieval.search").setLevel(logging.WARNING)
logging.getLogger("embeddings.embedder").setLevel(logging.WARNING)
logging.getLogger("llm.generator").setLevel(logging.WARNING)
logging.getLogger("config.settings").setLevel(logging.WARNING)
logging.getLogger("context.builder").setLevel(logging.WARNING)
logging.getLogger("ingestion.chunker").setLevel(logging.WARNING)
logging.getLogger("evaluation.corpus").setLevel(logging.WARNING)
logging.getLogger("ingestion.document_cluster").setLevel(logging.WARNING)
logging.getLogger("models.client").setLevel(logging.WARNING)
logging.getLogger("retrieval.reranker").setLevel(logging.WARNING)
logging.getLogger("verification.verifier").setLevel(logging.WARNING)


# ===========================================================================
# Phase 2 — Helpers
# ===========================================================================

_COLUMNS = [
    ("Config",          22, "left",  "config"),
    ("F1",               8, "right", "f1"),
    ("EM",               8, "right", "em"),
    ("Recall",           8, "right", "recall"),
    ("Hit@k",            8, "right", "hit_at_k"),
    ("AnsInCtx",         8, "right", "answer_in_context"),
    ("Faithful",         9, "right", "faithfulness"),
    ("Ret(ms)",          9, "right", "retrieval_ms"),
    ("Gen(ms)",          9, "right", "generation_ms"),
]


def fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def print_config_header(settings: Any) -> None:
    gen = settings.generator
    emb = settings.embedding
    ner = settings.ner
    ver = settings.verifier
    rer = settings.reranker
    db = settings.database

    print("=" * 78)
    print("  Hybrid-RAG — Single-Passage Evaluation")
    print("=" * 78)
    print()
    print("  Configuration:")
    print(f"    Generator:  {gen.model} @ {gen.base_url}")
    print(f"    Embedding:  {emb.api_model or emb.model} @ {emb.api_base_url}")
    print(f"    NER:        {ner.model} @ {ner.base_url}")
    print(f"    Verifier:   {ver.api_model} @ {ver.api_base_url}  (enabled={ver.enabled})")
    print(f"    Reranker:   {rer.api_model} @ {rer.api_base_url}  (backend={rer.backend})")
    print(f"    Database:   host={db.host} db={db.db_name} user={db.user}")
    print(f"    Top-K:      {_args.top_k}  |  Candidate-K:  {_args.candidate_k}")
    print(f"    Graph mode: {'enabled' if _args.with_graph else 'disabled'}")
    DB = _args.data_file or "selected_distractor_validation_n1_seed42.json"
    print(f"    Data file:  {DB}")
    print()


def load_records() -> list[dict]:
    """Load HotpotQA records from the cached JSON (or prepare+load)."""
    if _args.data_file:
        p = Path(_args.data_file)
        if not p.exists():
            print(f"  ERROR: data file not found: {p}", file=sys.stderr)
            sys.exit(1)
        return json.loads(p.read_text(encoding="utf-8"))

    # Default: cached n=1, seed=42
    from evaluation.dataset import load_cached
    try:
        return load_cached(n=1, seed=42)
    except FileNotFoundError:
        print("  Cache not found — preparing n=1, seed=42 ...")
        from evaluation.dataset import prepare_dataset
        return prepare_dataset(n=1, seed=42)


def run_retrieval_config(
    label: str,
    record: dict,
    query_emb: list[float],
    use_kwargs: dict,
) -> dict:
    """Run one retrieval configuration end-to-end, return all metrics."""
    question = record["question"]
    gold_answer = record["answer"]
    gold_titles = record["supporting_titles"]

    # --- Retrieval ---
    t0 = time.perf_counter()
    result = search(
        question,
        query_embedding=query_emb,
        top_k=_args.top_k,
        candidate_k=_args.candidate_k,
        **use_kwargs,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    retrieved_titles = [c.source_id for c in result.chunks]
    contexts = [c.text for c in result.chunks]
    if result.graph_facts:
        contexts.append(f"[Graph Facts]\n{result.graph_facts}")

    # --- Context assembly ---
    context_str, _citations = build_context(result.chunks, result.graph_facts)

    # --- Generation ---
    t1 = time.perf_counter()
    answer = generate(question, context_str, stream=False, eval_mode=True)
    generation_ms = (time.perf_counter() - t1) * 1000.0
    answer_str = str(answer) if answer else "None"

    # --- Answer metrics ---
    f1 = f1_score(answer_str, gold_answer)
    em_val = exact_match(answer_str, gold_answer)
    recall = token_recall(answer_str, gold_answer)

    # --- Retrieval metrics ---
    hit = retrieval_hit_at_k(retrieved_titles, gold_titles)
    aic = answer_in_context(gold_answer, contexts)

    # --- Faithfulness (verifier) ---
    faithfulness: float | None = None
    try:
        from verification.verifier import score_faithfulness
        faithfulness = score_faithfulness(answer_str, context_str)
    except Exception as exc:
        LOGGER.debug("Faithfulness verifier skipped: %s", exc)

    return {
        "config": label,
        "f1": f1,
        "em": em_val,
        "recall": recall,
        "hit_at_k": hit,
        "answer_in_context": aic,
        "faithfulness": faithfulness,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": retrieval_ms + generation_ms,
        "answer": answer_str,
        "gold": gold_answer,
        "retrieved_titles": retrieved_titles,
        "gold_titles": gold_titles,
    }


def run_direct(record: dict) -> dict:
    """Closed-book (no retrieval) baseline."""
    t0 = time.perf_counter()
    answer = generate(record["question"], closed_book=True, stream=False, eval_mode=True)
    gen_ms = (time.perf_counter() - t0) * 1000.0
    answer_str = str(answer) if answer else "None"

    f1 = f1_score(answer_str, record["answer"])
    em_val = exact_match(answer_str, record["answer"])
    recall = token_recall(answer_str, record["answer"])

    return {
        "config": "direct (closed-book)",
        "f1": f1,
        "em": em_val,
        "recall": recall,
        "hit_at_k": None,
        "answer_in_context": None,
        "faithfulness": None,
        "retrieval_ms": 0.0,
        "generation_ms": gen_ms,
        "total_ms": gen_ms,
        "answer": answer_str,
        "gold": record["answer"],
        "retrieved_titles": [],
        "gold_titles": record["supporting_titles"],
    }


def print_results_table(results: list[dict]) -> None:
    """Print the metrics table and retrieval details."""
    # --- Header ---
    parts_hdr: list[str] = []
    parts_sep: list[str] = []
    for title, width, align, _key in _COLUMNS:
        if align == "right":
            parts_hdr.append(title.rjust(width))
        else:
            parts_hdr.append(title.ljust(width))
        parts_sep.append("─" * width)
    print("  " + "  ".join(parts_hdr))
    print("  " + "  ".join(parts_sep))

    # --- Rows ---
    for r in results:
        cells: list[str] = []
        for title, width, align, key in _COLUMNS:
            v = r.get(key)
            if title == "Config":
                s = str(v).ljust(width)
            elif key.endswith("_ms") and v is not None:
                s = f"{v:.1f}".rjust(width)
            else:
                s = (fmt(v).rjust(width) if v is not None else "—".rjust(width))
            cells.append(s)
        print("  " + "  ".join(cells))

    print()

    # --- Retrieval details ---
    print("  ── Retrieved paragraphs ──")
    for r in results:
        label = r["config"]
        rt = r["retrieved_titles"]
        gt = r["gold_titles"]
        if not rt:
            print(f"  {label:<25} (no retrieval)")
        else:
            hits = [t for t in rt if t in gt]
            miss = [t for t in gt if t not in rt]
            print(f"  {label:<25} {rt}")
            if hits:
                print(f"  {'':<25} ✓ gold found: {hits}")
            if miss:
                print(f"  {'':<25} ✗ gold missed: {miss}")


# ===========================================================================
# Phase 3 — Main
# ===========================================================================

def main() -> None:
    settings = get_settings()
    print_config_header(settings)

    # ------------------------------------------------------------------
    # 3a — Load data
    # ------------------------------------------------------------------
    print("[1/5] Loading HotpotQA query ...")
    records = load_records()
    record = records[0]
    print(f"  Question:   {record['question']}")
    print(f"  Gold Answer: {record['answer']}")
    print(f"  Supporting:  {record['supporting_titles']}")
    n_paras = len(record["paragraphs"])
    n_gold = len(record["supporting_titles"])
    print(f"  Paragraphs:  {n_paras} ({n_gold} gold + {n_paras - n_gold} distractors)")
    print()

    # ------------------------------------------------------------------
    # 3b — DB schema
    # ------------------------------------------------------------------
    print("[2/5] Ensuring database schema ...")
    init_db()
    print("  OK")

    # Clean existing data so we start fresh
    if _args.clean:
        print("  Cleaning existing data from DB ...")
        import psycopg
        with psycopg.connect(settings.database.conninfo) as conn:
            conn.execute("DELETE FROM document_questions")
            conn.execute("DELETE FROM document_clusters")
            conn.execute("DELETE FROM chunks")
            conn.commit()
        print("  Done.")
    print()

    # ------------------------------------------------------------------
    # 3c — Ingest corpus
    # ------------------------------------------------------------------
    if not _args.skip_ingest:
        print("[3/5] Ingesting corpus into pgvector ...")
        corpus: dict[str, str] = {}
        for para in record["paragraphs"]:
            t = para["title"]
            if t not in corpus and para["text"]:
                corpus[t] = para["text"]
        print(f"  Corpus: {len(corpus)} unique paragraphs")

        counts = ingest_corpus(
            corpus,
            with_graph=_args.with_graph,
            backend="deepseek",
        )
        print(f"  Ingested: {counts['titles']} titles, {counts['chunks']} chunks"
              f"{', ' + str(counts['graph_triplets']) + ' graph triplets' if _args.with_graph else ''}")
    else:
        print("[3/5] Skipping ingestion (--skip-ingest)")
    print()

    # ------------------------------------------------------------------
    # 3d — Query embedding
    # ------------------------------------------------------------------
    print("[4/5] Computing query embedding ...")
    question = record["question"]
    query_emb = embedder([question])[0]
    print(f"  Embedding dimension: {len(query_emb)}")
    print()

    # ------------------------------------------------------------------
    # 3e — Run configurations
    # ------------------------------------------------------------------
    print("[5/5] Running retrieval + generation ⏳")
    print()

    # Define the configuration matrix
    configs: list[tuple[str, dict]] = [
        ("vector-only",        {"use_vector": True,  "use_bm25": False, "use_graph": False, "rerank": False}),
        ("vector+rerank",      {"use_vector": True,  "use_bm25": False, "use_graph": False, "rerank": True}),
        ("vector+BM25",        {"use_vector": True,  "use_bm25": True,  "use_graph": False, "rerank": False}),
        ("vector+BM25+rerank", {"use_vector": True,  "use_bm25": True,  "use_graph": False, "rerank": True}),
    ]

    if _args.with_graph:
        configs += [
            ("all_three",        {"use_vector": True,  "use_bm25": True,  "use_graph": True, "rerank": False}),
            ("all_three+rerank", {"use_vector": True,  "use_bm25": True,  "use_graph": True, "rerank": True}),
        ]

    results: list[dict] = []

    for label, kwargs in configs:
        print(f"  ── {label} ──")
        try:
            r = run_retrieval_config(label, record, query_emb, kwargs)
            results.append(r)
            print(f"      Answer:    {r['answer']}")
            print(f"      Gold:      {r['gold']}")
            print(f"      Retrieved: {r['retrieved_titles']}")
            print(f"      F1={r['f1']:.4f}  EM={r['em']:.4f}  Recall={r['recall']:.4f}  "
                  f"Hit@k={r['hit_at_k']:.4f}  AnsInCtx={r['answer_in_context']:.4f}")
            if r["faithfulness"] is not None:
                print(f"      Faithfulness: {r['faithfulness']:.4f}")
            print(f"      Latency: {r['retrieval_ms']:.0f}ms ret + "
                  f"{r['generation_ms']:.0f}ms gen = {r['total_ms']:.0f}ms")
        except Exception as exc:
            print(f"      FAILED: {exc}")
            import traceback
            traceback.print_exc()
        print()

    # If --direct-only, skip retrieval configs and just show closed-book
    if _args.direct_only:
        results = []

    # Closed-book baseline
    print(f"  ── direct (closed-book) ──")
    try:
        r = run_direct(record)
        results.append(r)
        print(f"      Answer:    {r['answer']}")
        print(f"      Gold:      {r['gold']}")
        print(f"      F1={r['f1']:.4f}  EM={r['em']:.4f}  Recall={r['recall']:.4f}")
        print(f"      Latency:   {r['generation_ms']:.0f}ms gen")
    except Exception as exc:
        print(f"      FAILED: {exc}")
    print()

    # ==================================================================
    # Phase 4 — Final Report
    # ==================================================================
    print()
    print("=" * 78)
    print("  FINAL RESULTS")
    print("=" * 78)
    print()

    if results:
        print_results_table(results)
        print()

        # Best retrieval config (by F1)
        retrieval_results = [r for r in results if r["config"] != "direct (closed-book)"]
        if retrieval_results:
            best = max(retrieval_results, key=lambda r: r["f1"])
            print(f"  🏆  Best retrieval config:  {best['config']}")
            print(f"      F1={best['f1']:.4f}  EM={best['em']:.4f}  "
                  f"Hit@k={best['hit_at_k']:.4f}  AnsInCtx={best['answer_in_context']:.4f}")
    else:
        print("  (no results to show)")

    print()
    print(f"  Query: {record['question']}")
    print(f"  Gold:  {record['answer']}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
