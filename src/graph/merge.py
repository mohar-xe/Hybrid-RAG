"""Embedding-similarity node merging for the knowledge graph.

This is an *explicit*, user-invoked maintenance step. Run it AFTER triplets
have been upserted into Kùzu (see ``retrieval.kuzu_store.upsert_triplets``);
it never runs automatically during ingestion.

Two entities are treated as "near-same" when the cosine similarity of their
name embeddings is at or above ``settings.graph.merge_similarity_threshold``
(default 0.90). Merging folds the duplicate's relationships (both directions)
and its chunk mentions onto a single canonical node, keeping the strongest
weight for any duplicated relation. Deduplicating keeps the weight meaningful
for both retrieval (weight = confidence) and visualization (weight = node
proximity: the higher the weight, the closer two nodes are drawn).

Candidate detection is **ANN-based** for scalability. Entity-name embeddings
are persisted in a (non-indexed) ``Entity.embedding`` column — computed once
per entity, since the name is the primary key — and near-duplicates are found
with an in-memory HNSW index (``hnswlib``). That is ~O(N log N) instead of the
O(N²) all-pairs comparison, and repeat runs only embed newly added entities.
If ``hnswlib`` is unavailable, it transparently falls back to the exact O(N²)
search.

Typical flow::

    from graph.merge import find_merge_candidates, merge_similar_nodes

    # 1. review what *would* merge (read-only)
    for c in find_merge_candidates():
        print(c.drop, "->", c.keep, round(c.similarity, 3))

    # 2. once happy, apply
    merge_similar_nodes(apply=True)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.settings import get_settings
from embeddings.embedder import embedder
from retrieval.kuzu_store import get_connection
from constants.logger import setup_logger
from constants.exceptions import GraphError

LOGGER = setup_logger(__name__)
settings = get_settings()

# In-memory HNSW (hnswlib) parameters for ANN candidate search.
_HNSW_M = 16              # graph connectivity (higher = better recall, more memory)
_EF_CONSTRUCTION = 200    # build-time accuracy/speed trade-off
_MIN_EF = 64              # query-time search breadth (must be >= neighbors)
_DEFAULT_NEIGHBORS = 10   # nearest neighbours examined per entity


@dataclass(frozen=True)
class MergeCandidate:
    """A near-duplicate entity pair proposed for merging."""

    keep: str          # canonical node that survives the merge
    drop: str          # duplicate node folded into ``keep``
    similarity: float  # cosine similarity of the two name embeddings


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _ensure_embedding_column(conn, dim: int) -> None:
    """Make sure the (non-indexed) ``Entity.embedding`` column exists.

    Stored as a fixed-size FLOAT vector. We deliberately do NOT put a Kùzu
    vector index on this column: Kùzu forbids ``SET`` on an indexed column,
    which would block incremental embedding. ANN search is done in-memory via
    hnswlib instead.
    """
    conn.execute(f"ALTER TABLE Entity ADD IF NOT EXISTS embedding FLOAT[{dim}]")


def _embed_missing_entities(conn, dim: int) -> int:
    """Embed and persist names that don't have an embedding yet.

    Entity names are the primary key and never change, so each embedding is
    computed only once; subsequent runs embed only newly added entities.
    Returns the number of entities embedded.
    """
    missing = [
        row[0]
        for row in conn.execute(
            "MATCH (e:Entity) WHERE e.embedding IS NULL RETURN e.name ORDER BY e.name"
        )
    ]
    if not missing:
        return 0

    vectors = embedder(
        missing,
        model=settings.embedding.model,
        dim=dim,
        batch_size=settings.embedding.batch_size,
    )
    for name, vec in zip(missing, vectors):
        conn.execute(
            "MATCH (e:Entity {name: $n}) SET e.embedding = $v",
            {"n": name, "v": [float(x) for x in vec]},
        )
    LOGGER.info(f"Embedded {len(missing)} new entity name(s) for merge search.")
    return len(missing)


def _degree(conn, name: str) -> int:
    """Total (in + out) RELATES_TO degree of an entity."""
    res = conn.execute(
        "MATCH (e:Entity {name: $n})-[r:RELATES_TO]-(:Entity) RETURN count(r)",
        {"n": name},
    )
    for row in res:
        return int(row[0])
    return 0


def _pick_canonical(conn, names: set[str]) -> str:
    """Choose the surviving node: highest degree, then shortest, then A–Z."""
    return sorted(names, key=lambda n: (-_degree(conn, n), len(n), n))[0]


def _cosine_candidate_pairs(
    names: list[str], vectors: np.ndarray, threshold: float
) -> list[tuple[str, str, float]]:
    """Pure similarity step: return ``(a, b, sim)`` for cosine >= ``threshold``.

    Kept free of any DB/embedding I/O so it can be unit-tested directly with
    synthetic vectors.
    """
    if len(names) < 2:
        return []

    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = vectors / norms
    sims = unit @ unit.T

    pairs: list[tuple[str, str, float]] = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s >= threshold:
                pairs.append((names[i], names[j], round(s, 4)))

    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def _ann_candidate_pairs(
    names: list[str], vectors: np.ndarray, threshold: float, k: int
) -> list[tuple[str, str, float]]:
    """ANN near-duplicate search via an in-memory HNSW index (hnswlib).

    Builds the index once and batch-queries every point's ``k`` nearest
    neighbours — ~O(N log N) build + O(N·k·log N) query — then keeps pairs with
    cosine similarity >= ``threshold``. Falls back to the exact O(N²) search if
    hnswlib is not installed.
    """
    n = len(names)
    if n < 2:
        return []

    try:
        import hnswlib
    except ImportError:
        LOGGER.warning(
            "hnswlib unavailable; falling back to brute-force O(N^2) similarity."
        )
        return _cosine_candidate_pairs(names, vectors, threshold)

    vectors = np.asarray(vectors, dtype=np.float32)
    dim = int(vectors.shape[1])

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=_EF_CONSTRUCTION, M=_HNSW_M)
    index.add_items(vectors, np.arange(n))
    index.set_ef(max(k + 1, _MIN_EF))

    # +1 because each point's own nearest neighbour is itself.
    labels, distances = index.knn_query(vectors, k=min(k + 1, n))

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for lbl, dist in zip(labels[i], distances[i]):
            j = int(lbl)
            if j == i:
                continue
            sim = 1.0 - float(dist)  # hnswlib 'cosine' distance = 1 - cosine_sim
            if sim >= threshold:
                key = (i, j) if i < j else (j, i)
                if key not in seen:
                    seen.add(key)
                    pairs.append((names[i], names[j], round(sim, 4)))

    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def find_merge_candidates(
    threshold: float | None = None,
    *,
    neighbors: int = _DEFAULT_NEIGHBORS,
    conn=None,
) -> list[MergeCandidate]:
    """Detect near-duplicate entities via ANN over persisted name embeddings.

    Does not change graph *structure* (no nodes/edges added or removed), but it
    will persist an embedding for any entity that doesn't have one yet. Returns
    candidates sorted by descending similarity so the user can review before
    deciding to merge. ``neighbors`` controls how many nearest neighbours per
    entity are examined by the ANN search.
    """
    if threshold is None:
        threshold = settings.graph.merge_similarity_threshold
    if conn is None:
        _, conn = get_connection()

    dim = settings.embedding.embed_dim
    _ensure_embedding_column(conn, dim)
    try:
        _embed_missing_entities(conn, dim)
    except Exception as e:
        raise GraphError(f"Failed to embed entity names for merge: {e}") from e

    rows = [
        (row[0], row[1])
        for row in conn.execute(
            "MATCH (e:Entity) WHERE e.embedding IS NOT NULL "
            "RETURN e.name, e.embedding ORDER BY e.name"
        )
    ]
    if len(rows) < 2:
        return []

    names = [r[0] for r in rows]
    vectors = np.asarray([list(r[1]) for r in rows], dtype=np.float32)

    candidates: list[MergeCandidate] = []
    for a, b, sim in _ann_candidate_pairs(names, vectors, threshold, neighbors):
        keep = _pick_canonical(conn, {a, b})
        drop = b if keep == a else a
        candidates.append(MergeCandidate(keep=keep, drop=drop, similarity=sim))

    LOGGER.info(
        f"Found {len(candidates)} merge candidate(s) at threshold {threshold:.2f} "
        f"(ANN over {len(names)} entities)."
    )
    return candidates


def merge_nodes(keep: str, drop: str, *, conn=None) -> None:
    """Fold entity ``drop`` into ``keep`` and delete ``drop``.

    Relationships in both directions and chunk mentions are redirected onto
    ``keep``. When ``keep`` already has a matching relation, the higher weight
    (confidence) is retained. Self-loops are skipped.
    """
    if keep == drop:
        return
    if conn is None:
        _, conn = get_connection()

    # Materialise edges first — never mutate the graph while iterating a result.
    out_edges = [
        (row[0], row[1], row[2])
        for row in conn.execute(
            "MATCH (d:Entity {name: $drop})-[r:RELATES_TO]->(t:Entity) "
            "RETURN t.name, r.relation_type, r.weight",
            {"drop": drop},
        )
    ]
    in_edges = [
        (row[0], row[1], row[2])
        for row in conn.execute(
            "MATCH (s:Entity)-[r:RELATES_TO]->(d:Entity {name: $drop}) "
            "RETURN s.name, r.relation_type, r.weight",
            {"drop": drop},
        )
    ]
    chunk_ids = [
        row[0]
        for row in conn.execute(
            "MATCH (d:Entity {name: $drop})-[:MENTIONED_IN]->(c:Chunk) "
            "RETURN c.chunk_id",
            {"drop": drop},
        )
    ]

    for target, rel, weight in out_edges:
        if target == keep:  # would become a self-loop
            continue
        conn.execute(
            """
            MATCH (k:Entity {name: $keep}), (t:Entity {name: $target})
            MERGE (k)-[r:RELATES_TO {relation_type: $rel}]->(t)
            ON CREATE SET r.weight = $w
            ON MATCH SET r.weight = CASE WHEN r.weight < $w THEN $w ELSE r.weight END
            """,
            {"keep": keep, "target": target, "rel": rel, "w": weight},
        )

    for source, rel, weight in in_edges:
        if source == keep:  # would become a self-loop
            continue
        conn.execute(
            """
            MATCH (s:Entity {name: $source}), (k:Entity {name: $keep})
            MERGE (s)-[r:RELATES_TO {relation_type: $rel}]->(k)
            ON CREATE SET r.weight = $w
            ON MATCH SET r.weight = CASE WHEN r.weight < $w THEN $w ELSE r.weight END
            """,
            {"source": source, "keep": keep, "rel": rel, "w": weight},
        )

    for cid in chunk_ids:
        conn.execute(
            """
            MATCH (k:Entity {name: $keep}), (c:Chunk {chunk_id: $cid})
            MERGE (k)-[:MENTIONED_IN]->(c)
            """,
            {"keep": keep, "cid": cid},
        )

    conn.execute("MATCH (d:Entity {name: $drop}) DETACH DELETE d", {"drop": drop})
    LOGGER.info(
        f"Merged '{drop}' -> '{keep}' "
        f"({len(out_edges)} out, {len(in_edges)} in, {len(chunk_ids)} mention(s))."
    )


def merge_similar_nodes(
    threshold: float | None = None,
    apply: bool = False,
    *,
    neighbors: int = _DEFAULT_NEIGHBORS,
) -> list[MergeCandidate]:
    """Find near-duplicate entities and, when ``apply`` is True, merge them.

    With ``apply=False`` (default) this is a dry run: it returns the candidate
    pairs only, leaving the graph untouched so the user can decide. With
    ``apply=True`` transitive duplicates (A~B, B~C) are grouped via union-find
    and each group collapses into a single canonical node.
    """
    _, conn = get_connection()
    candidates = find_merge_candidates(threshold, neighbors=neighbors, conn=conn)
    if not apply or not candidates:
        return candidates

    # Union-find so transitive near-duplicates collapse into one canonical node.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for c in candidates:
        union(c.keep, c.drop)

    groups: dict[str, set[str]] = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)

    merged_groups = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = _pick_canonical(conn, members)
        for member in members:
            if member != canonical:
                merge_nodes(canonical, member, conn=conn)
        merged_groups += 1

    LOGGER.info(
        f"Applied merges across {merged_groups} group(s) "
        f"from {len(candidates)} candidate pair(s)."
    )
    return candidates
