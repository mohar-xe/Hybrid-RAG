"""All-three retrieval smoke pipeline on DUMMY data (no full eval, no live stack).

The HotpotQA corpus is already ingested for the real benchmark, but running the
100-question matrix (or even one live query) needs PostgreSQL + Ollama + the
generator endpoint. This script instead proves the **all-three** retrieval path
— dense (vector) + sparse (BM25) + the new **multi-hop** knowledge graph, fused
and assembled into a prompt — works end to end on small, deterministic dummy
data, so the wiring can be validated in CI / offline.

How it stays offline while still exercising the *real* pipeline:

* dense + sparse retrieval are replaced with fixed dummy ranked lists (the parts
  that would otherwise hit pgvector / Postgres), but the **RRF fusion** that
  combines them is the real ``retrieval.search.search`` code path;
* the graph signal runs the **real** ``get_entity_context`` multi-hop BFS against
  a throwaway Kùzu graph seeded in a temp dir (no touching the project graph);
* the **real** ``context.builder.build_context`` assembles the final prompt;
* generation is intentionally *not* run (that needs the live LLM) — the script
  stops at the assembled context, which is exactly the all-three retrieval
  product.

Run it::

    uv run python -m evaluation.smoke_all_three        # from the repo root

Exit code is 0 on success and 1 if any check fails, so it doubles as a smoke
test in addition to printing a readable per-signal breakdown.
"""

from __future__ import annotations

import shutil
import sys
import tempfile

import evaluation  # noqa: F401  -- runs the src/ path shim in evaluation/__init__.py

QUESTION = "How are Transformer and Translation related?"

# Dummy dense/sparse rankings. d1 & d2 appear in BOTH lists (so RRF should float
# them to the top); d3/d5 are dense-only and d4 is sparse-only, so a correct
# fusion keeps signal from each retriever rather than just one.
_DOCS = {
    "d1": (
        "Transformer",
        "The Transformer architecture introduced the self-attention mechanism.",
    ),
    "d2": (
        "Attention",
        "Attention mechanisms are widely used for machine translation tasks.",
    ),
    "d3": ("BERT", "BERT is a language model based on the Transformer encoder."),
    "d4": (
        "RNN",
        "Recurrent networks were the predecessor approach used before attention.",
    ),
    "d5": (
        "Seq2Seq",
        "Sequence-to-sequence models map an input sequence to an output sequence.",
    ),
}
_DENSE_ORDER = ["d1", "d3", "d2", "d5"]  # "semantic" ranking
_SPARSE_ORDER = ["d2", "d1", "d4"]  # "keyword" ranking


def _make_chunks(order):
    """Build real ``RetrievedChunk`` objects for a dummy ranking (high score first)."""
    from retrieval.pgvector import RetrievedChunk

    n = len(order)
    out = []
    for rank, cid in enumerate(order):
        source_id, text = _DOCS[cid]
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                text=text,
                source_id=source_id,
                score=float(n - rank),
                source_type="DUMMY",
                chunk_index=rank,
            )
        )
    return out


def _seed_graph(conn):
    """Seed the multi-hop chain: Transformer -> Attention -> Translation -> Task.

    ``Attention used_for Translation`` is the 2-hop bridge from ``Transformer`` —
    the fact a single-hop lookup can never reach.
    """
    from graph.schema import Node, Relation, Triplet
    from retrieval.kuzu_store import init_graph_schema, upsert_triplets

    def t(src, rel, w, dst):
        return Triplet(
            source=Node(title=src, type="concept"),
            relation=Relation(type=rel, weight=w),
            target=Node(title=dst, type="concept"),
        )

    init_graph_schema(conn=conn)
    upsert_triplets(
        [
            t("Transformer", "introduces", 0.95, "Attention"),
            t("Attention", "used_for", 0.92, "Translation"),
            t("BERT", "based_on", 0.90, "Transformer"),
            t("Translation", "is_type_of", 0.80, "Task"),
        ],
        conn=conn,
    )


class _Checks:
    """Tiny pass/fail accumulator so the script reads as a checklist."""

    def __init__(self):
        self.failed = 0

    def __call__(self, label: str, ok: bool):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            self.failed += 1


def main() -> int:
    import kuzu

    import retrieval.search as search_module
    from context.builder import build_context
    from retrieval.kuzu_store import get_entity_context as real_get_entity_context
    from retrieval.search import extract_query_entities, search

    tmpdir = tempfile.mkdtemp(prefix="hgr_smoke_")
    db = conn = None
    check = _Checks()
    try:
        # --- Stand up the dummy graph in a throwaway Kùzu DB --------------------
        db = kuzu.Database(f"{tmpdir}/graph_db")
        conn = kuzu.Connection(db)
        _seed_graph(conn)

        # --- Swap the live retrievers for dummy data --------------------------
        # search() binds vector_search/bm25_search/embedder/get_entity_context in
        # its own module namespace, so patching there reroutes the real fusion
        # code onto dummy inputs without a database or Ollama.
        search_module.vector_search = lambda emb, top_k=20: _make_chunks(_DENSE_ORDER)
        search_module.bm25_search = lambda q, top_k=20: _make_chunks(_SPARSE_ORDER)
        search_module.embedder = lambda texts: [[0.0] * 8 for _ in texts]
        search_module.get_entity_context = lambda names, *a, **k: (
            real_get_entity_context(names, *a, **{**k, "conn": conn})
        )

        print("=" * 72)
        print("ALL-THREE RETRIEVAL SMOKE PIPELINE (dense + sparse + multi-hop graph)")
        print("=" * 72)
        print(f"Question: {QUESTION}\n")

        # --- Show each signal in isolation ------------------------------------
        print("Dense (vector) ranking:   ", _DENSE_ORDER)
        print("Sparse (BM25) ranking:    ", _SPARSE_ORDER)
        entities = extract_query_entities(QUESTION)
        print("Graph seed entities:      ", entities)

        # The multi-hop difference, made explicit on a single seed.
        one_hop = real_get_entity_context(["Transformer"], hops=1, conn=conn)
        two_hop = real_get_entity_context(["Transformer"], hops=2, conn=conn)
        print("\nGraph from 'Transformer' @ 1 hop (single-hop, old behavior):")
        print("   " + (one_hop.replace("\n", "\n   ") or "(none)"))
        print("Graph from 'Transformer' @ 2 hops (multi-hop, new behavior):")
        print("   " + (two_hop.replace("\n", "\n   ") or "(none)"))

        bridge = "Attention used_for Translation"

        # --- Run the real all-three retrieval ---------------------------------
        result = search(
            QUESTION,
            use_vector=True,
            use_bm25=True,
            use_graph=True,
            rerank=False,  # cross-encoder skipped to stay offline
            top_k=5,
            query_embedding=[0.0] * 8,  # provided so embedder is never needed
        )
        fused_ids = [c.chunk_id for c in result.chunks]

        print("\nFused (RRF) chunk order:  ", fused_ids)
        print("Graph facts (all three):")
        print("   " + (result.graph_facts.replace("\n", "\n   ") or "(none)"))

        # --- Assemble the final prompt context (real builder) -----------------
        context, citations = build_context(result.chunks, result.graph_facts)
        print(f"\nAssembled context: {len(citations)} chunk citation(s) + graph block")
        print("-" * 72)
        print(context)
        print("-" * 72)

        # --- Checks -----------------------------------------------------------
        print("\nChecks:")
        check("multi-hop is live: 1-hop misses the bridge fact", bridge not in one_hop)
        check("multi-hop is live: 2-hop surfaces the bridge fact", bridge in two_hop)
        check(
            "dense + sparse fused (both d1 & d2, the shared docs, ranked top-2)",
            set(fused_ids[:2]) == {"d1", "d2"},
        )
        check(
            "fusion kept a dense-only doc (d3 or d5)",
            bool({"d3", "d5"} & set(fused_ids)),
        )
        check("fusion kept the sparse-only doc (d4)", "d4" in fused_ids)
        check(
            "graph signal contributed multi-hop bridge fact to all-three result",
            bridge in result.graph_facts,
        )
        check(
            "final context embeds the graph knowledge block",
            "[Graph Knowledge]" in context,
        )
        check("final context embeds retrieved chunk text", _DOCS["d1"][1] in context)

        print()
        if check.failed:
            print(f"RESULT: {check.failed} check(s) FAILED.")
            return 1
        print(
            "RESULT: all checks passed — dense + sparse + multi-hop graph fuse correctly."
        )
        return 0
    finally:
        # Drop references before removing the on-disk DB.
        conn = None
        db = None
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
