"""K-Means clustering + medoids over stored chunk embeddings (Stage 2).

An explicit, user-invoked indexing step (run via the ``reindex`` CLI command,
mirroring ``merge-graph``) — never automatic during ingestion. It partitions the
stored chunk embeddings into ``K = max(3, round(sqrt(n)))`` clusters and marks,
per cluster, the medoid: the actual chunk whose embedding is closest (cosine) to
the cluster centroid. ``cluster_id`` and ``is_medoid`` are persisted on every
chunk and drive the coarse->fine retrieval in Stage 3.

Embeddings are L2-normalized at ingest, so cosine similarity is a dot product
and we use spherical K-Means (assign by max dot product, re-normalize centroids).
"""

from dataclasses import dataclass

import numpy as np
import psycopg

from config.settings import get_settings
from constants.logger import setup_logger
from constants.exceptions import DatabaseError

LOGGER = setup_logger(__name__)
settings = get_settings()


@dataclass
class ReindexResult:
    n_chunks: int
    k: int
    cluster_sizes: dict[int, int]
    n_medoids: int


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _fetch_embeddings(conn) -> tuple[list[str], np.ndarray]:
    rows = conn.execute("SELECT chunk_id, embedding::text FROM chunks ORDER BY chunk_id").fetchall()
    ids = [r[0] for r in rows]
    if not rows:
        return ids, np.empty((0, 0), dtype=np.float32)
    vecs = np.array(
        [np.fromstring(r[1].strip("[]"), sep=",", dtype=np.float32) for r in rows],
        dtype=np.float32,
    )
    return ids, vecs


def _kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> np.ndarray:
    """Spherical K-Means on L2-normalized rows. Returns per-row cluster labels."""
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.full(len(X), -1, dtype=int)

    for _ in range(iters):
        sims = X @ centroids.T            # cosine similarity (rows are unit-norm)
        new_labels = sims.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members):
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids[c] = centroid / norm if norm > 0 else centroids[c]
            else:
                # Re-seed an empty cluster from a random point to keep K stable.
                centroids[c] = X[rng.integers(len(X))]
    return labels


def _medoid_indices(X: np.ndarray, labels: np.ndarray, k: int) -> set[int]:
    """Index of the medoid (max cosine to its cluster centroid) per cluster."""
    medoids: set[int] = set()
    for c in range(k):
        member_idx = np.flatnonzero(labels == c)
        if member_idx.size == 0:
            continue
        members = X[member_idx]
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        best_local = (members @ centroid).argmax()
        medoids.add(int(member_idx[best_local]))
    return medoids


def reindex(conn_info: str | None = None) -> ReindexResult:
    """Cluster all stored chunks and persist ``cluster_id`` + ``is_medoid``.

    Raises ``DatabaseError`` if there are fewer than 3 chunks (clustering is not
    meaningful below K=3).
    """
    conninfo = conn_info or settings.database.conninfo
    with psycopg.connect(conninfo) as conn:
        ids, X = _fetch_embeddings(conn)
        n = len(ids)
        if n < 3:
            raise DatabaseError(
                f"Need at least 3 chunks to cluster (found {n}). Ingest more first."
            )

        X = _l2_normalize(X)
        k = max(3, round(n ** 0.5))
        k = min(k, n)  # never more clusters than points

        labels = _kmeans(X, k)
        medoids = _medoid_indices(X, labels, k)

        with conn.cursor() as cur:
            # Reset prior assignment, then write the new one.
            cur.execute("UPDATE chunks SET cluster_id = NULL, is_medoid = FALSE")
            cur.executemany(
                "UPDATE chunks SET cluster_id = %s, is_medoid = %s WHERE chunk_id = %s",
                [(int(labels[i]), i in medoids, ids[i]) for i in range(n)],
            )
        conn.commit()

    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    LOGGER.info(f"Reindexed {n} chunks into {k} clusters; {len(medoids)} medoids.")
    return ReindexResult(n_chunks=n, k=k, cluster_sizes=sizes, n_medoids=len(medoids))
