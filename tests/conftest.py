"""Shared pytest fixtures for GraphBench tests.

Provides:
- sample_triples: a small set of (subject, relation, object) tuples.
- sample_triple_dicts: same triples as Triple TypedDicts.
- sample_questions: a few HotpotQA-style question/answer dicts.
- tiny_embeddings: 5 random L2-normalised float32 embeddings for FAISS tests.
- mock_neo4j_client: Neo4jClient with driver mocked out (no live DB needed).

"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graphbench.ingestion import Triple


@pytest.fixture
def sample_triples() -> list[tuple[str, str, str]]:
    """Return a small list of knowledge graph triples for unit tests."""
    return [
        ("Albert Einstein", "born_in", "Ulm"),
        ("Albert Einstein", "field_of_work", "Physics"),
        ("Ulm", "located_in", "Germany"),
        ("Marie Curie", "born_in", "Warsaw"),
        ("Marie Curie", "field_of_work", "Chemistry"),
        ("Warsaw", "located_in", "Poland"),
    ]


@pytest.fixture
def sample_triple_dicts() -> list[Triple]:
    """Return a small list of Triple TypedDicts for ingestion unit tests."""
    return [
        {"subject": "albert einstein", "relation": "place_of_birth", "object": "ulm"},
        {
            "subject": "albert einstein",
            "relation": "field_of_work",
            "object": "physics",
        },
        {"subject": "ulm", "relation": "country", "object": "germany"},
        {"subject": "marie curie", "relation": "place_of_birth", "object": "warsaw"},
        {"subject": "marie curie", "relation": "field_of_work", "object": "chemistry"},
        {"subject": "warsaw", "relation": "country", "object": "poland"},
    ]


@pytest.fixture
def sample_questions() -> list[dict]:
    """Return a small list of HotpotQA-style question dicts for unit tests."""
    return [
        {
            "id": "q1",
            "question": "Where was Albert Einstein born?",
            "answer": "Ulm",
            "type": "bridge",
        },
        {
            "id": "q2",
            "question": "What field did Marie Curie work in?",
            "answer": "Chemistry",
            "type": "comparison",
        },
    ]


@pytest.fixture
def tiny_embeddings() -> tuple[list[str], np.ndarray]:
    """Return 5 random L2-normalised entity embeddings for FAISS tests.

    Returns:
        Tuple of (entity_strings, embeddings) where embeddings has shape (5, 384)
        with each row L2-normalised to unit length.
    """
    rng = np.random.default_rng(42)
    entities = ["alpha", "beta", "gamma", "delta", "epsilon"]
    raw = rng.standard_normal((5, 384)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return entities, raw / norms


@pytest.fixture
def mock_neo4j_client():
    """Return a Neo4jClient with the neo4j Driver mocked out.

    The mock prevents any real network connections. Use this for testing
    methods that call execute_write / execute_read without a live AuraDB.
    """
    with patch("graphbench.utils.neo4j_client.GraphDatabase.driver") as mock_driver:
        mock_driver_instance = MagicMock()
        mock_driver.return_value = mock_driver_instance

        # execute_read returns empty list by default; override in specific tests
        mock_session = MagicMock()
        mock_driver_instance.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_driver_instance.session.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_session.execute_read.return_value = []
        mock_session.execute_write.return_value = None

        from graphbench.utils.neo4j_client import Neo4jClient

        client = Neo4jClient(uri="bolt://mock:7687", username="neo4j", password="test")
        yield client
