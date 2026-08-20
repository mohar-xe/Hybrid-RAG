"""Report writer for fast_eval — real latency, plus the LLM-call tally."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation import metrics
from evaluation.fast_eval import config
from evaluation.fast_eval.runner import CellResult

_COLUMNS = [
    ("config", "Config"),
    ("rerank", "Rerank"),
    ("f1", "F1"),
    ("em", "EM"),
    ("recall", "Recall"),
    ("hit_at_k", "Hit@k (top)"),
    ("answer_in_context", "AnsInCtx"),
    ("latency_mean_ms", "Lat mean (ms)"),
    ("latency_median_ms", "Lat med (ms)"),
    ("latency_p95_ms", "Lat p95 (ms)"),
    ("retrieval_mean_ms", "Ret mean (ms)"),
    ("generation_mean_ms", "Gen mean (ms)"),
]


def _aggregate(cells: list[CellResult]) -> dict:
    f1 = metrics.mean([c.f1 for c in cells])
    em = metrics.mean([c.em for c in cells])
    recall = metrics.mean([c.recall for c in cells])
    hits = [c.hit_at_k for c in cells if c.hit_at_k is not None]
    aics = [c.answer_in_context for c in cells if c.answer_in_context is not None]
    return {
        "f1": f1,
        "em": em,
        "recall": recall,
        "hit_at_k": metrics.mean(hits) if hits else None,
        "answer_in_context": metrics.mean(aics) if aics else None,
        "latency": metrics.aggregate_latency([c.total_ms for c in cells]),
        "retrieval_latency": metrics.aggregate_latency([c.retrieval_ms for c in cells]),
        "generation_latency": metrics.aggregate_latency([c.generation_ms for c in cells]),
    }


def build_results(cells: list[CellResult]) -> list[dict]:
    """Group cells by spec, aggregate, and flatten into report rows."""
    by_label: dict[str, list[CellResult]] = {}
    order: list[str] = []
    for c in cells:
        if c.spec.label not in by_label:
            by_label[c.spec.label] = []
            order.append(c.spec.label)
        by_label[c.spec.label].append(c)

    rows = []
    for label in order:
        group = by_label[label]
        spec = group[0].spec
        agg = _aggregate(group)
        rows.append(
            {
                "mode": spec.mode,
                "rerank": spec.rerank,
                "n": len(group),
                "f1": agg["f1"],
                "em": agg["em"],
                "recall": agg["recall"],
                "hit_at_k": agg["hit_at_k"],
                "answer_in_context": agg["answer_in_context"],
                "latency": agg["latency"],
                "retrieval_latency": agg["retrieval_latency"],
                "generation_latency": agg["generation_latency"],
            }
        )
    return rows


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def to_markdown(rows: list[dict], meta: dict) -> str:
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = ["# Hybrid-RAG — HotpotQA Evaluation (fast/paid)", ""]
    lines.append(
        f"- **Queries:** {meta.get('n')}  •  **Seed:** {meta.get('seed')}  •  "
        f"**top_k:** {meta.get('top_k')}  •  **Workers:** {meta.get('max_workers')}  •  "
        f"**Generated:** {meta.get('generated_at')}"
    )
    lines.append(
        f"- **LLM calls:** {meta.get('llm_calls_total')} total "
        f"(gen={meta.get('generation_calls')}, entity={meta.get('entity_calls')}, "
        f"embedding={meta.get('embedding_calls')}, rerank={meta.get('rerank_calls')})"
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lat = r["latency"]
        rlat = r["retrieval_latency"]
        glat = r["generation_latency"]
        cells = [
            r["mode"],
            "yes" if r["rerank"] else "no",
            _fmt(r["f1"]),
            _fmt(r["em"]),
            _fmt(r["recall"]),
            _fmt(r["hit_at_k"]),
            _fmt(r["answer_in_context"]),
            f"{lat['mean_ms']:.0f}",
            f"{lat['median_ms']:.0f}",
            f"{lat['p95_ms']:.0f}",
            f"{rlat['mean_ms']:.0f}",
            f"{glat['mean_ms']:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    rows: list[dict],
    meta: dict,
    *,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Write markdown + JSON to fast_eval's own results dir. Returns (md, json)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = stem or f"fast_eval_{timestamp}"
    if "generated_at" not in meta:
        meta = {**meta, "generated_at": timestamp}
    meta = {**meta, "staged": False}

    md_path = config.RESULTS_DIR / f"{stem}.md"
    md_path.write_text(to_markdown(rows, meta), encoding="utf-8")

    json_path = config.RESULTS_DIR / f"{stem}.json"
    json_path.write_text(
        json.dumps({"meta": meta, "results": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    return md_path, json_path
