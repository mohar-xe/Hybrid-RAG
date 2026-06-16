"""Reproducible HotpotQA evaluation harness for Hybrid-RAG.

Compares four retrieval configurations — ``direct`` (closed-book, no retrieval),
``semantic`` (dense only), ``semantic_bm25`` (dense + sparse, RRF-fused), and
``all_three`` (dense + sparse + graph) — each with reranking on and off, and
reports F1, EM, answer recall, retrieval hit@k ("top"), and latency.

Run from the repo root::

    uv run --extra eval python -m evaluation.run_eval --help

Reproducibility: the 100 evaluation queries are chosen by a seeded RNG over the
HotpotQA validation split and cached to ``evaluation/data/`` as JSON, so every
run (and every machine) evaluates exactly the same questions.
"""

import sys
from pathlib import Path

# The project uses a src-layout; the uv editable install already exposes
# ``config``/``retrieval``/``llm``/... on sys.path. This shim makes the harness
# also work under a bare ``python`` invocation (no-op when already importable).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
