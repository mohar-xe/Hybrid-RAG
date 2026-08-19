"""Aggregate run results into a human-readable markdown table.

Pure standard-library; consumes the ``RunResult`` dictionaries produced by
``runner`` so it can be tested without any heavy dependencies.

Latency caveat: the eval runs in decoupled, cache-persisted phases (retrieval
and generation are separate stages) and generation is bundled (~40
question/context pairs per API call). Per-query latency is therefore **not a
meaningful metric** here — the latency columns render ``n/a`` and raw timings
are kept only in the sibling ``.json`` results file for reference.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Column order for the markdown table.
_COLUMNS = [
    ("config", "Config"),
    ("rerank", "Rerank"),
    ("f1", "F1"),
    ("em", "EM"),
    ("recall", "Recall"),
    ("hit_at_k", "Hit@k (top)"),
    ("answer_in_context", "AnsInCtx"),
    ("graph_lift", "GraphLift"),
    ("latency_mean_ms", "Latency mean (ms)"),
    ("latency_p95_ms", "Latency p95 (ms)"),
]

# Always included in the report: per-query latency is meaningless in the
# staged/batched pipeline.
LATENCY_CAVEAT = (
    "> **Latency caveat:** this evaluation runs in decoupled, cache-persisted "
    "phases (artifacts → retrieve → generate) and generation bundles ~40 "
    "(question, context) pairs per API call for cost efficiency. Per-query "
    "latency is therefore **not comparable** to an interactive RAG system: "
    "retrieval ran in a separate phase from generation, and generation time "
    "was amortized across bundled calls. Latency columns render `n/a`; raw "
    "timings are preserved in the companion `.json` results file for reference."
)


def _round_floats(obj: object, decimals: int = 4) -> object:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


def _fmt(value, decimals: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _flatten(result: dict) -> dict:
    """Flatten a RunResult dict into a single row for tabular output.

    Latency fields are always ``n/a`` (see module docstring / LATENCY_CAVEAT);
    raw timings live in the companion JSON results file.
    """
    return {
        "config": result["mode"],
        "rerank": "yes" if result["rerank"] else "no",
        "f1": _round_floats(result["f1"]),
        "em": _round_floats(result["em"]),
        "recall": _round_floats(result["recall"]),
        "hit_at_k": _round_floats(result["hit_at_k"]),
        "answer_in_context": _round_floats(result["answer_in_context"]),
        "graph_lift": _round_floats(result.get("graph_lift")),
        "latency_mean_ms": "n/a",
        "latency_p95_ms": "n/a",
    }


def to_markdown(results: list[dict], meta: dict | None = None) -> str:
    rows = [_flatten(r) for r in results]
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _ in _COLUMNS:
            decimals = 1 if key.startswith("latency") else 4
            cells.append(_fmt(row.get(key), decimals))
        lines.append("| " + " | ".join(cells) + " |")

    preamble = ["# Hybrid-RAG — HotpotQA Evaluation", ""]
    if meta:
        preamble.append(
            f"- **Queries:** {meta.get('n')}  •  **Seed:** {meta.get('seed')}  •  "
            f"**top_k:** {meta.get('top_k')}  •  **Generated:** {meta.get('generated_at')}"
        )
        if meta.get("graph_ingested") is False:
            preamble.append(
                "- _Note: graph was not ingested, so `all_three` graph facts were empty._"
            )
        preamble.append("")
        if meta.get("staged"):
            preamble.append(LATENCY_CAVEAT)
            preamble.append("")
    return "\n".join(preamble + lines) + "\n"


def _json_dumps(results: list[dict], meta: dict | None = None) -> str:
    """Serialize results (+ meta, raw latency timings included) as pretty JSON."""
    payload = {"meta": meta or {}, "results": results}
    return json.dumps(payload, indent=2, default=str)


def write_report(
    results: list[dict],
    out_dir: Path,
    *,
    meta: dict | None = None,
    stem: str | None = None,
) -> Path:
    """Write a markdown report to ``out_dir``. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = stem or f"eval_{timestamp}"

    meta = dict(meta or {})
    meta.setdefault("generated_at", timestamp)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(to_markdown(results, meta), encoding="utf-8")

    return md_path
