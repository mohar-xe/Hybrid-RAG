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


def get_entity_context(
    entity_names: list[str],
    hops: int = 2,
    min_weight: float | None = None,
    *,
    conn=None,
) -> str:
    """Return relation facts for the given entities, filtered by confidence.

    Only relations whose weight is strictly greater than ``min_weight``
    (default: ``settings.graph.min_relation_weight`` = 0.50) are returned;
    lower-weight relations are treated as low-confidence noise and dropped.
    Facts are ordered highest-confidence first.
    """
    if min_weight is None:
        min_weight = settings.graph.min_relation_weight

    if conn is None:
        _, conn = get_connection()
    facts: list[str] = []
    seen: set[str] = set()

    for name in entity_names:
        result = conn.execute(
            """
            MATCH (e1:Entity {name: $name})-[r:RELATES_TO]->(e2:Entity)
            WHERE r.weight > $min_weight
            RETURN e1.name, r.relation_type, e2.name, r.weight
            ORDER BY r.weight DESC
            LIMIT 10
            """,
            {"name": name, "min_weight": min_weight},
        )
        for row in result:
            fact = f"{row[0]} {row[1]} {row[2]}"
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)

    return "\n".join(facts)