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
from retrieval.kuzu_store import get_connection, init_graph_schema
from constants.logger import setup_logger
from constants.exceptions import GraphError

LOGGER = setup_logger(__name__)
settings = get_settings()

# In-memory HNSW (hnswlib) parameters for ANN candidate search.
_HNSW_M = 16              # graph connectivity (higher = better recall, more memory)
_EF_CONSTRUCTION = 200    # build-time accuracy/speed trade-off
_MIN_EF = 64              # query-time search breadth (must be >= neighbors)
_DEFAULT_NEIGHBORS = 10   # nearest neighbours examined per entity
# Minimum character-trigram Jaccard overlap required *in addition* to embedding
# cosine before two names are merged. Short entity names embed with a high
# cosine floor (anisotropy), so cosine alone matches unrelated names; requiring
# real surface overlap removes those false positives. Tune if it under-merges.
_MIN_LEXICAL_JACCARD = 0.2


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


def _lexical_similarity(a: str, b: str) -> float:
    """Character-trigram Jaccard overlap of two names (case/space-insensitive).

    A cheap, language-agnostic surface-similarity signal used to *gate* embedding
    matches. nomic-embed-text vectors of short strings sit on a high cosine floor
    (anisotropy), so cosine >= 0.90 alone pairs unrelated names like 'MLP' with
    'KANs', or 'Michael Griebel' with 'KANs'. Requiring genuine string overlap as
    well removes those cross-domain false positives, while real variants
    ('B-spline'/'splines', 'X theorem'/'generalized X theorem') still pass.

    Trade-off: acronym<->expansion pairs ('KAN'/'Kolmogorov-Arnold Network') are
    NOT caught by this gate. That is accepted on purpose — a missed merge is
    cheap and re-runnable; an over-merge silently destroys distinct entities.
    """
    def _trigrams(s: str) -> set[str]:
        s = " " + " ".join(s.lower().split()) + " "
        return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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

    # Ensure the graph schema exists first. On a fresh clone (or after running
    # only `ingest --no-graph`), opening the Kùzu DB creates an *empty* database
    # with no `Entity` table, so the ALTER in `_ensure_embedding_column` would
    # raise "Binder exception: Table Entity does not exist". `init_graph_schema`
    # is idempotent (CREATE ... IF NOT EXISTS): a populated graph is untouched,
    # and an empty one degrades cleanly to "no merge candidates".
    init_graph_schema(conn=conn)

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
        # Gate the embedding match on real surface overlap: cosine alone is not
        # discriminative for short names (anisotropy floor ~0.90), so without
        # this, unrelated names like 'MLP'/'KANs' become "duplicates".
        if _lexical_similarity(a, b) < _MIN_LEXICAL_JACCARD:
            continue
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
    pairs only, leaving the graph untouched so the user can decide.

    With ``apply=True`` duplicates are merged **without transitive chaining**.
    Candidate pairs are processed highest-similarity first and a canonical node
    *absorbs* its duplicates, but a node that has already been merged away can
    never become a survivor, and a surviving node can never be folded away. This
    deliberately replaces single-linkage union-find: entity names embed with a
    high cosine floor, so a few false-positive pairs under union-find can bridge
    unrelated clusters (people, methods, citations) into one giant component that
    collapses onto the highest-degree node. Absorb-only merging bounds the blast
    radius of any stray false positive to a single fold. True multi-spelling
    duplicates that are each similar to the same canonical still all collapse
    onto it; chains whose endpoints are not themselves similar do not.
    """
    _, conn = get_connection()
    candidates = find_merge_candidates(threshold, neighbors=neighbors, conn=conn)
    if not apply or not candidates:
        return candidates

    # Roles are sticky: a 'keep' stays a survivor and a 'drop' stays merged-away,
    # so no node is ever both -- which is exactly what prevents A->B->C chaining.
    role: dict[str, str] = {}
    applied = 0
    for c in candidates:  # already sorted by similarity, highest first
        if role.get(c.drop) is not None:   # drop already merged (as survivor or dropped)
            continue
        if role.get(c.keep) == "drop":     # keep was merged away -> merging here would chain
            continue
        merge_nodes(c.keep, c.drop, conn=conn)
        role[c.keep] = "keep"
        role[c.drop] = "drop"
        applied += 1

    LOGGER.info(
        f"Applied {applied} merge(s) from {len(candidates)} candidate pair(s) "
        f"(absorb-only, no transitive chaining)."
    )
    return candidates
