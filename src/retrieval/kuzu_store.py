"""KùzuDB graph storage and multi-hop retrieval."""

import kuzu
from pathlib import Path

from config.settings import get_settings
from graph.schema import Triplet
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()


def get_connection() -> tuple[kuzu.Database, kuzu.Connection]:
    """Open the Kùzu database and return (database, connection).

    Public so other graph modules (e.g. ``graph.merge``) can reuse a single
    connection instead of opening their own handle to the same DB path.
    """
    db_path = str(settings.graph.db_path)
    # Kùzu 0.11.x stores the database as a single file, so create the *parent*
    # directory only — creating a directory at db_path itself makes Kùzu raise
    # "Database path cannot be a directory".
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    return db, conn


def init_graph_schema(*, conn=None) -> None:
    if conn is None:
        _, conn = get_connection()

    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Entity (
            name STRING PRIMARY KEY,
            entity_type STRING
        )
    """)
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Chunk (
            chunk_id STRING PRIMARY KEY,
            text STRING,
            source_id STRING
        )
    """)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS RELATES_TO (
            FROM Entity TO Entity,
            relation_type STRING,
            weight DOUBLE
        )
    """)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS MENTIONED_IN (
            FROM Entity TO Chunk
        )
    """)
    LOGGER.info("Graph schema initialized.")


def upsert_triplets(triplets: list[Triplet], *, conn=None) -> None:
    if conn is None:
        _, conn = get_connection()

    for t in triplets:
        conn.execute(
            """
            MERGE (n:Entity {name: $name})
            ON CREATE SET n.entity_type = $type
            ON MATCH SET n.entity_type = $type
            """,
            {"name": t.source.title, "type": t.source.type},
        )
        conn.execute(
            """
            MERGE (n:Entity {name: $name})
            ON CREATE SET n.entity_type = $type
            ON MATCH SET n.entity_type = $type
            """,
            {"name": t.target.title, "type": t.target.type},
        )
        conn.execute(
            """
            MATCH (a:Entity {name: $src})
            MATCH (b:Entity {name: $dst})
            MERGE (a)-[:RELATES_TO {relation_type: $rel, weight: $w}]->(b)
            """,
            {"src": t.source.title, "dst": t.target.title, "rel": t.relation.type, "w": t.relation.weight},
        )
    LOGGER.info(f"Upserted {len(triplets)} triplets.")


def link_entities_to_chunk(
    entity_names: list[str],
    chunk_id: str,
    text: str = "",
    source_id: str = "",
    *,
    conn=None,
) -> None:
    """Create MENTIONED_IN edges from entities to a chunk.

    The Chunk node is MERGE-d here first: nothing else inserts Chunk nodes, so
    without this the entity->chunk MATCH would find no chunk and the links would
    silently never form. ``text``/``source_id`` populate the node on creation.
    """
    if conn is None:
        _, conn = get_connection()

    conn.execute(
        """
        MERGE (c:Chunk {chunk_id: $cid})
        ON CREATE SET c.text = $text, c.source_id = $sid
        """,
        {"cid": chunk_id, "text": text, "sid": source_id},
    )
    for name in entity_names:
        conn.execute(
            """
            MATCH (e:Entity {name: $name})
            MATCH (c:Chunk {chunk_id: $cid})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            {"name": name, "cid": chunk_id},
        )


def _one_hop_neighbors(
    conn,
    name: str,
    min_weight: float,
    limit: int,
    both_directions: bool,
) -> list[tuple[str, str, str, float, str]]:
    """Weight-filtered one-hop neighbors of ``name`` (the BFS expansion step).

    Returns ``(subject, relation, object, weight, neighbor)`` rows where
    ``neighbor`` is the entity on the *other* end of the edge — the node to
    enqueue for the next hop. Queries the outgoing direction and, unless
    ``both_directions`` is False, the incoming direction too: edge direction in
    an extracted KG is largely an artifact of how a sentence was phrased
    ("BERT based_on Transformer" vs. "Transformer predecessor_of BERT"), so a
    seed that only appears as a relation's *target* would otherwise yield no
    facts and dead-end the traversal. ``limit`` bounds the fan-out per direction.
    """
    limit = max(1, int(limit))
    rows: list[tuple[str, str, str, float, str]] = []

    # Matching is case-INSENSITIVE (lower(e.name) = lower($name)) rather than an
    # exact PK lookup: query seeds come from YAKE/LLM extraction and rarely carry
    # the exact casing the entity was stored under ("general relativity" vs the
    # stored "General Relativity"). A side benefit on a graph with case-variant
    # duplicates is that one seed now reaches *all* variants at once. The trade
    # is a label scan instead of a PK probe — fine for this embedded, modest graph.

    # Outgoing: (name)-[r]->(other). neighbor == object.
    outgoing = conn.execute(
        "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) "
        "WHERE lower(e1.name) = lower($name) AND r.weight > $min_weight "
        "RETURN e1.name, r.relation_type, e2.name, r.weight "
        "ORDER BY r.weight DESC "
        "LIMIT " + str(limit),
        {"name": name, "min_weight": min_weight},
    )
    for row in outgoing:
        rows.append((row[0], row[1], row[2], row[3], row[2]))

    if both_directions:
        # Incoming: (other)-[r]->(name). neighbor == subject.
        incoming = conn.execute(
            "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) "
            "WHERE lower(e2.name) = lower($name) AND r.weight > $min_weight "
            "RETURN e1.name, r.relation_type, e2.name, r.weight "
            "ORDER BY r.weight DESC "
            "LIMIT " + str(limit),
            {"name": name, "min_weight": min_weight},
        )
        for row in incoming:
            rows.append((row[0], row[1], row[2], row[3], row[0]))

    return rows


def get_entity_context(
    entity_names: list[str],
    hops: int | None = None,
    min_weight: float | None = None,
    *,
    max_facts: int | None = None,
    per_hop_neighbors: int | None = None,
    both_directions: bool = True,
    conn=None,
) -> str:
    """Return **multi-hop** relation facts for the given seed entities.

    Breadth-first traversal of the ``RELATES_TO`` graph out to ``hops`` edges
    from each seed entity (default ``GRAPH__MAX_HOPS`` = 2), collecting the
    relation facts encountered along the way. This is what makes the graph a
    *multi-hop* retriever rather than a lookup: a single hop only returns a
    seed's direct relations, but "bridge" questions need a chain — seed ``A`` →
    ``A rel B`` (hop 1) → ``B rel C`` (hop 2) — where ``C`` is reachable only by
    following the bridge entity ``B``. Both facts land in the context so the
    generator can connect them.

    Only relations with weight strictly greater than ``min_weight`` (default
    ``GRAPH__MIN_RELATION_WEIGHT`` = 0.50) are followed or returned; weaker
    relations are treated as low-confidence noise. Traversal is bidirectional by
    default, fans out at most ``per_hop_neighbors`` edges per node per hop, and
    stops once ``max_facts`` distinct facts are gathered. Facts are returned
    closest-hop-first (a seed's direct relations before the bridges they reach),
    and duplicates (including a fact re-encountered from the other endpoint) are
    dropped. ``hops``/``min_weight``/``max_facts``/``per_hop_neighbors`` fall
    back to their ``GRAPH__*`` settings when left as ``None``.
    """
    if hops is None:
        hops = settings.graph.max_hops
    if min_weight is None:
        min_weight = settings.graph.min_relation_weight
    if max_facts is None:
        max_facts = settings.graph.max_facts
    if per_hop_neighbors is None:
        per_hop_neighbors = settings.graph.per_hop_neighbors

    if conn is None:
        _, conn = get_connection()

    facts: list[str] = []
    # All de-duplication is case-insensitive (lowercased keys) to mirror the
    # case-insensitive node matching: otherwise case-variant nodes
    # ("X networks" vs "X Networks") would re-expand and emit near-duplicate
    # facts. Display strings keep their original (first-seen) casing.
    seen_facts: set[str] = set()       # lowercased "subj rel obj" keys
    visited: set[str] = set()          # lowercased entity names already expanded
    queued: set[str] = set()           # lowercased names already enqueued
    # BFS frontier: seed entities to expand first (de-duplicated case-insensitively,
    # order-preserved, empties dropped). Each hop replaces it with new neighbors.
    frontier: list[str] = []
    for n in dict.fromkeys(entity_names):
        if n and n.lower() not in queued:
            frontier.append(n)
            queued.add(n.lower())

    for _hop in range(max(1, hops)):
        if not frontier or len(facts) >= max_facts:
            break
        next_frontier: list[str] = []
        for name in frontier:
            if name.lower() in visited:
                continue
            visited.add(name.lower())
            for subj, rel, obj, _weight, neighbor in _one_hop_neighbors(
                conn, name, min_weight, per_hop_neighbors, both_directions
            ):
                fact = f"{subj} {rel} {obj}"
                fact_key = fact.lower()
                if fact_key not in seen_facts:
                    seen_facts.add(fact_key)
                    facts.append(fact)
                    if len(facts) >= max_facts:
                        break
                if neighbor.lower() not in visited and neighbor.lower() not in queued:
                    next_frontier.append(neighbor)
                    queued.add(neighbor.lower())
            if len(facts) >= max_facts:
                break
        frontier = next_frontier

    return "\n".join(facts)