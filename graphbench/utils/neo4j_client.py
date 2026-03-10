"""Neo4j AuraDB client for GraphBench.

Thin wrapper around the official neo4j-python-driver with:
- Connection pooling (max 10 connections).
- Automatic retry on TransientError (up to 3 attempts).
- Context-manager support for safe session lifecycle management.
- High-level helpers: subgraph extraction, entity lookup, schema setup.

All Cypher for generic read/write lives here. Module-specific query templates
(e.g. the UNWIND MERGE pattern in neo4j_writer.py) belong in their callers.
"""

import logging
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import TransientError

from graphbench.utils.config import settings

logger = logging.getLogger(__name__)

_MAX_RETRY = 3


class Neo4jClient:
    """Thread-safe Neo4j AuraDB client with automatic retry.

    Usage (context manager — preferred):
        with Neo4jClient() as client:
            client.execute_write(cypher, name="Alice")

    Usage (manual lifecycle):
        client = Neo4jClient()
        client.execute_write(cypher, name="Alice")
        client.close()
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialise the driver (does not open a session yet).

        Args:
            uri: Neo4j connection URI. Defaults to settings.neo4j_uri.
            username: Neo4j username. Defaults to settings.neo4j_username.
            password: Neo4j password. Defaults to settings.neo4j_password.
        """
        self._uri = uri or settings.neo4j_uri
        self._driver: Driver = GraphDatabase.driver(
            self._uri,
            auth=(
                username or settings.neo4j_username,
                password or settings.neo4j_password,
            ),
            max_connection_pool_size=10,
        )
        logger.info("Neo4jClient initialised for URI: %s", self._uri)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the driver and release all pooled connections."""
        self._driver.close()
        logger.info("Neo4jClient closed.")

    def verify_connectivity(self) -> None:
        """Ping the server to verify the connection is alive.

        Raises:
            ServiceUnavailable: If the server cannot be reached.
        """
        self._driver.verify_connectivity()
        logger.info("Neo4j connectivity verified.")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create uniqueness constraint on Entity.name if not already present.

        Safe to call multiple times (uses IF NOT EXISTS). Must be called before
        bulk triple insertion to make MERGE queries efficient.
        """
        cypher = (
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        with self._driver.session() as session:
            session.run(cypher)
        logger.info("Schema constraint 'entity_name_unique' ensured.")

    # ------------------------------------------------------------------
    # Generic read / write
    # ------------------------------------------------------------------

    def execute_write(self, cypher: str, **params: Any) -> None:
        """Run a write Cypher query with auto-retry on TransientError.

        Args:
            cypher: Cypher query string using $param_name placeholders.
            **params: Keyword arguments passed as Cypher parameters.

        Raises:
            TransientError: If all retry attempts are exhausted.
        """

        def _tx(tx: Any) -> None:
            tx.run(cypher, **params)

        for attempt in range(1, _MAX_RETRY + 1):
            try:
                with self._driver.session() as session:
                    session.execute_write(_tx)
                return
            except TransientError as exc:
                if attempt == _MAX_RETRY:
                    raise
                logger.warning(
                    "TransientError on attempt %d/%d: %s. Retrying...",
                    attempt,
                    _MAX_RETRY,
                    exc,
                )

    def execute_read(self, cypher: str, **params: Any) -> list[dict]:
        """Run a read Cypher query and return results as a list of dicts.

        Args:
            cypher: Cypher query string using $param_name placeholders.
            **params: Keyword arguments passed as Cypher parameters.

        Returns:
            List of result records as plain Python dicts.
        """

        def _tx(tx: Any) -> list[dict]:
            result = tx.run(cypher, **params)
            return [record.data() for record in result]

        with self._driver.session() as session:
            return session.execute_read(_tx)

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    def get_subgraph(
        self,
        entity_name: str,
        *,
        hops: int | None = None,
        directed: bool = False,
    ) -> list[tuple[str, str, str]]:
        """Extract a k-hop subgraph centred on an entity.

        Args:
            entity_name: Name of the seed entity node.
            hops: Number of relationship hops. Defaults to settings.subgraph_hops (2).
            directed: If True, traverse relationships in their stored direction only.
                If False (default), traverse in both directions.

        Returns:
            List of (subject, relation, object) tuples within k hops.
            Subject and object are entity name strings.
            Relation is the Neo4j relationship type string.
        """
        k = hops if hops is not None else settings.subgraph_hops
        rel_pattern = "-[*1..$k]->" if directed else "-[*1..$k]-"
        cypher = f"""
        MATCH path = (seed:Entity {{name: $name}}){rel_pattern}(neighbor:Entity)
        UNWIND relationships(path) AS rel
        RETURN
            startNode(rel).name AS subject,
            type(rel)           AS relation,
            endNode(rel).name   AS object
        """
        results = self.execute_read(cypher, name=entity_name, k=k)
        return [(r["subject"], r["relation"], r["object"]) for r in results]

    def get_subgraph_multi(
        self,
        entity_names: list[str],
        *,
        hops: int | None = None,
    ) -> list[tuple[str, str, str]]:
        """Extract and merge subgraphs for multiple seed entities.

        Args:
            entity_names: List of seed entity names.
            hops: Number of hops. Defaults to settings.subgraph_hops.

        Returns:
            Deduplicated list of (subject, relation, object) tuples.
        """
        seen: set[tuple[str, str, str]] = set()
        triples: list[tuple[str, str, str]] = []
        for name in entity_names:
            for triple in self.get_subgraph(name, hops=hops):
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)
        return triples

    def find_entity(self, name: str) -> dict | None:
        """Look up a single entity by exact name.

        Args:
            name: Entity name to look up.

        Returns:
            Dict with entity properties {"name": str}, or None if not found.
        """
        cypher = "MATCH (e:Entity {name: $name}) RETURN e.name AS name LIMIT 1"
        results = self.execute_read(cypher, name=name)
        return results[0] if results else None

    def count_triples(self) -> int:
        """Return total number of relationship triples stored in the graph.

        Returns:
            Integer count of all relationships in the database.
        """
        results = self.execute_read("MATCH ()-[r]->() RETURN count(r) AS count")
        return results[0]["count"] if results else 0
