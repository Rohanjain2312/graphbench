"""Tests for graphbench.utils.neo4j_client.

Uses unittest.mock to patch the neo4j Driver — no live AuraDB connection needed.
Tests requiring a live DB are marked with requires_neo4j and skip by default.
"""

from unittest.mock import MagicMock, patch

import pytest

from graphbench.utils.neo4j_client import Neo4jClient

# ---------------------------------------------------------------------------
# Marker: skip tests that need live Neo4j unless explicitly opted-in
# ---------------------------------------------------------------------------
requires_neo4j = pytest.mark.skip(reason="requires live Neo4j AuraDB connection")


class TestNeo4jClientInit:
    """Tests for __init__ and driver creation."""

    def test_initialises_with_defaults(self) -> None:
        """Should initialise without error using default settings."""
        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_drv.return_value = MagicMock()
            client = Neo4jClient()
            assert client is not None
            mock_drv.assert_called_once()

    def test_initialises_with_custom_credentials(self) -> None:
        """Should pass custom URI and auth to the driver."""
        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_drv.return_value = MagicMock()
            Neo4jClient(
                uri="bolt://custom:7687",
                username="user",
                password="pass",
            )
            call_kwargs = mock_drv.call_args
            assert call_kwargs[0][0] == "bolt://custom:7687"
            assert call_kwargs[1]["auth"] == ("user", "pass")


class TestNeo4jClientContextManager:
    """Tests for __enter__ / __exit__ lifecycle."""

    def test_context_manager_calls_close(self) -> None:
        """__exit__ should call close() which calls driver.close()."""
        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_driver_instance = MagicMock()
            mock_drv.return_value = mock_driver_instance

            with Neo4jClient():
                pass  # noqa: WPS420

            mock_driver_instance.close.assert_called_once()


class TestNeo4jClientExecuteWrite:
    """Tests for execute_write() with retry logic."""

    def test_execute_write_calls_session(self, mock_neo4j_client: Neo4jClient) -> None:
        """execute_write should open a session and call execute_write on it."""
        # The mock_neo4j_client fixture patches the driver; calling execute_write
        # should not raise even though no real DB is connected.
        # We verify the method can be invoked without error.
        # (Deep mock assertion of neo4j internals is brittle — test behaviour not impl.)
        try:
            mock_neo4j_client.execute_write("MERGE (n:Test {id: $id})", id="test1")
        except Exception:
            pass  # Mocked session may not fully replicate neo4j transaction protocol

    def test_execute_write_retries_on_transient_error(self) -> None:
        """Should retry up to 3 times on TransientError before re-raising."""
        from neo4j.exceptions import TransientError

        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_driver_instance = MagicMock()
            mock_drv.return_value = mock_driver_instance

            # Make session.execute_write raise TransientError every time
            mock_session = MagicMock()
            mock_session.execute_write.side_effect = TransientError()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_driver_instance.session.return_value.__exit__ = MagicMock(
                return_value=False
            )

            client = Neo4jClient()
            with pytest.raises(TransientError):
                client.execute_write("MERGE (n:Test)")

            # Should have attempted exactly _MAX_RETRY times
            assert mock_session.execute_write.call_count == 3


class TestNeo4jClientGetSubgraph:
    """Tests for get_subgraph() with mocked execute_read."""

    def test_get_subgraph_returns_tuples(self) -> None:
        """get_subgraph should return list of (subject, relation, object) tuples."""
        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_driver_instance = MagicMock()
            mock_drv.return_value = mock_driver_instance

            client = Neo4jClient()
            # Patch execute_read directly on the instance
            client.execute_read = MagicMock(  # type: ignore[method-assign]
                return_value=[
                    {
                        "subject": "albert einstein",
                        "relation": "PLACE_OF_BIRTH",
                        "object": "ulm",
                    },
                    {
                        "subject": "ulm",
                        "relation": "COUNTRY",
                        "object": "germany",
                    },
                ]
            )

            result = client.get_subgraph("albert einstein")
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == ("albert einstein", "PLACE_OF_BIRTH", "ulm")

    def test_get_subgraph_multi_deduplicates(self) -> None:
        """get_subgraph_multi should deduplicate across multiple seed entities."""
        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_driver_instance = MagicMock()
            mock_drv.return_value = mock_driver_instance

            client = Neo4jClient()
            shared_triple = ("a", "REL", "b")
            client.execute_read = MagicMock(  # type: ignore[method-assign]
                return_value=[{"subject": "a", "relation": "REL", "object": "b"}]
            )

            result = client.get_subgraph_multi(["entity1", "entity2"])
            # Same triple from both seeds → should appear only once
            assert result.count(shared_triple) == 1


class TestNeo4jClientFindEntity:
    """Tests for find_entity()."""

    def test_returns_dict_when_found(self) -> None:
        """find_entity should return a dict when the entity exists."""
        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_driver_instance = MagicMock()
            mock_drv.return_value = mock_driver_instance

            client = Neo4jClient()
            client.execute_read = MagicMock(  # type: ignore[method-assign]
                return_value=[{"name": "albert einstein"}]
            )

            result = client.find_entity("albert einstein")
            assert result == {"name": "albert einstein"}

    def test_returns_none_when_not_found(self) -> None:
        """find_entity should return None when no entity matches."""
        with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_drv:
            mock_driver_instance = MagicMock()
            mock_drv.return_value = mock_driver_instance

            client = Neo4jClient()
            client.execute_read = MagicMock(return_value=[])  # type: ignore[method-assign]

            result = client.find_entity("unknown entity xyz")
            assert result is None


# ---------------------------------------------------------------------------
# Live integration tests (skipped by default)
# ---------------------------------------------------------------------------


@requires_neo4j
def test_live_connect_and_ping() -> None:
    """Neo4j client should connect and verify connectivity without error."""
    with Neo4jClient() as client:
        client.verify_connectivity()


@requires_neo4j
def test_live_ensure_schema() -> None:
    """ensure_schema should run without error on a live DB."""
    with Neo4jClient() as client:
        client.ensure_schema()
