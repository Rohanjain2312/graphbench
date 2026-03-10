"""Neo4j triple writer for the ingestion pipeline.

Bulk-inserts (subject, relation, object) triples into Neo4j AuraDB using
batched UNWIND MERGE Cypher (batch_size=500) to avoid duplicates.

Neo4j constraint: relationship types cannot be parameterized in Cypher.
Triples are therefore grouped by relationship type, and a separate
UNWIND MERGE query is generated per type. This is the correct idiomatic
pattern for dynamic relationship types in Neo4j.

Node label: Entity (property: name, with uniqueness constraint)
Relationship type: UPPER_SNAKE_CASE derived from relation string
"""

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

from graphbench.ingestion import Triple

if TYPE_CHECKING:
    from graphbench.utils.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


def _rel_type(relation: str) -> str:
    """Convert a relation string to a valid Neo4j relationship type label.

    Uppercases and replaces spaces and hyphens with underscores.

    Args:
        relation: snake_case or space-separated relation string.

    Returns:
        UPPER_SNAKE_CASE string safe for Neo4j relationship type literals.

    Examples:
        >>> _rel_type("place_of_birth")
        'PLACE_OF_BIRTH'
        >>> _rel_type("field of work")
        'FIELD_OF_WORK'
    """
    return relation.upper().replace(" ", "_").replace("-", "_")


def write_triples(
    triples: list[Triple],
    client: "Neo4jClient",
    *,
    batch_size: int = _BATCH_SIZE,
) -> int:
    """Batch-insert triples into Neo4j using idempotent UNWIND MERGE Cypher.

    Groups triples by relationship type (since Neo4j does not support
    parameterized relationship types). For each type, sends batches of
    (subject, object) pairs using UNWIND for bulk efficiency.

    Idempotent: running twice will not create duplicate nodes or edges.

    Args:
        triples: List of Triple dicts to insert.
        client: Connected Neo4jClient instance with schema constraints applied.
        batch_size: Number of (subject, object) pairs per transaction.

    Returns:
        Total number of triples successfully written.
    """
    if not triples:
        logger.warning("write_triples called with empty triple list. No-op.")
        return 0

    # Group by relationship type (required because Neo4j rel types are literals)
    by_rel: dict[str, list[dict[str, str]]] = {}
    for t in triples:
        rel = _rel_type(t["relation"])
        by_rel.setdefault(rel, []).append(
            {"subject": t["subject"], "object": t["object"]}
        )

    total_written = 0
    for rel_type, pairs in tqdm(by_rel.items(), desc="Writing relation types"):
        # Build dynamic Cypher template for this relation type
        # The f-string rel_type is safe here: derived from _rel_type() which
        # only produces UPPER_SNAKE_CASE alphanumeric strings.
        cypher = f"""
        UNWIND $pairs AS pair
        MERGE (a:Entity {{name: pair.subject}})
        MERGE (b:Entity {{name: pair.object}})
        MERGE (a)-[:{rel_type}]->(b)
        """

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            client.execute_write(cypher, pairs=batch)
            total_written += len(batch)
            logger.debug(
                "Wrote batch of %d triples for rel_type=%s (offset=%d)",
                len(batch),
                rel_type,
                i,
            )

    logger.info("Total triples written to Neo4j: %d", total_written)
    return total_written
