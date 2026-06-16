"""Aggregate run results into JSON, CSV, and a human-readable markdown table.

Pure standard-library; consumes the ``RunResult`` dictionaries produced by
``runner`` so it can be tested without any heavy dependencies.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# Column order for the flat CSV / markdown table.
_COLUMNS = [
    ("config", "Config"),
    ("rerank", "Rerank"),
    ("f1", "F1"),
    ("em", "EM"),
    ("recall", "Recall"),
    ("hit_at_k", "Hit@k (top)"),
    ("answer_in_context", "AnsInCtx"),
    ("latency_mean_ms", "Latency mean (ms)"),
    ("latency_p95_ms", "Latency p95 (ms)"),
]


def _fmt(value, decimals: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _flatten(result: dict) -> dict:
    """Flatten a RunResult dict into a single row for tabular output."""
    latency = result.get("latency", {}) or {}
    return {
        "config": result["mode"],
        "rerank": "yes" if result["rerank"] else "no",
        "f1": result["f1"],
        "em": result["em"],
        "recall": result["recall"],
        "hit_at_k": result["hit_at_k"],
        "answer_in_context": result["answer_in_context"],
        "latency_mean_ms": latency.get("mean_ms"),
        "latency_p95_ms": latency.get("p95_ms"),
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
    return "\n".join(preamble + lines) + "\n"


def write_reports(
    results: list[dict],
    out_dir: Path,
    *,
    meta: dict | None = None,
    stem: str | None = None,
) -> dict[str, Path]:
    """Write JSON + CSV + markdown reports to ``out_dir``. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = stem or f"eval_{timestamp}"

    meta = dict(meta or {})
    meta.setdefault("generated_at", timestamp)

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8"
    )

    rows = [_flatten(r) for r in results]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[key for key, _ in _COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(to_markdown(results, meta), encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "markdown": md_path}
