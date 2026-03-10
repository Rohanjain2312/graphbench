"""Tests for graphbench.ingestion modules.

Covers: rebel_loader, triple_extractor, embedder, faiss_writer, neo4j_writer.
All tests that require live services (Neo4j, HuggingFace downloads) are
either mocked or marked slow/skip.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from graphbench.ingestion import Triple
from graphbench.ingestion.rebel_loader import TOP_50_RELATIONS, stream_triples
from graphbench.ingestion.triple_extractor import filter_by_relation, parse_rebel_output

# ---------------------------------------------------------------------------
# rebel_loader tests
# ---------------------------------------------------------------------------


class TestStreamTriplesPreextracted:
    """Tests for stream_triples() with a pre-extracted parquet file."""

    def test_streams_from_parquet(self, tmp_path: Path) -> None:
        """Should yield Triple dicts from a valid parquet file."""
        parquet_path = tmp_path / "triples.parquet"
        df = pd.DataFrame(
            [
                {
                    "subject": "albert einstein",
                    "relation": "place_of_birth",
                    "object": "ulm",
                },
                {
                    "subject": "marie curie",
                    "relation": "field_of_work",
                    "object": "chemistry",
                },
                {
                    "subject": "ignored",
                    "relation": "unknown_relation_xyz",
                    "object": "also_ignored",
                },
            ]
        )
        df.to_parquet(parquet_path, index=False)

        results = list(stream_triples(preextracted_path=parquet_path))

        assert len(results) == 2, "unknown_relation_xyz should be filtered out"
        assert all(r["relation"] in TOP_50_RELATIONS for r in results)

    def test_streams_from_json(self, tmp_path: Path) -> None:
        """Should yield Triple dicts from a valid JSON file."""
        json_path = tmp_path / "triples.json"
        data = [
            {"subject": "ulm", "relation": "country", "object": "germany"},
        ]
        json_path.write_text(json.dumps(data))

        results = list(stream_triples(preextracted_path=json_path))
        assert len(results) == 1
        assert results[0]["subject"] == "ulm"

    def test_respects_max_triples(self, tmp_path: Path) -> None:
        """max_triples should cap the number of yielded triples."""
        parquet_path = tmp_path / "triples.parquet"
        df = pd.DataFrame(
            [
                {
                    "subject": f"entity_{i}",
                    "relation": "country",
                    "object": "somewhere",
                }
                for i in range(20)
            ]
        )
        df.to_parquet(parquet_path, index=False)

        results = list(stream_triples(preextracted_path=parquet_path, max_triples=5))
        assert len(results) == 5

    def test_filters_to_top50_relations(self, tmp_path: Path) -> None:
        """Only triples with TOP_50_RELATIONS should be yielded."""
        parquet_path = tmp_path / "triples.parquet"
        df = pd.DataFrame(
            [
                {"subject": "a", "relation": "country", "object": "b"},
                {"subject": "a", "relation": "not_in_top50_xyz", "object": "b"},
            ]
        )
        df.to_parquet(parquet_path, index=False)

        results = list(stream_triples(preextracted_path=parquet_path))
        assert all(r["relation"] in TOP_50_RELATIONS for r in results)
        assert len(results) == 1

    def test_normalises_entities(self, tmp_path: Path) -> None:
        """Entity strings should be lowercased and whitespace-collapsed."""
        parquet_path = tmp_path / "triples.parquet"
        df = pd.DataFrame(
            [
                {
                    "subject": "Albert  Einstein",
                    "relation": "place_of_birth",
                    "object": "ULM",
                },
            ]
        )
        df.to_parquet(parquet_path, index=False)

        results = list(stream_triples(preextracted_path=parquet_path))
        assert results[0]["subject"] == "albert einstein"
        assert results[0]["object"] == "ulm"

    def test_raises_on_missing_columns(self, tmp_path: Path) -> None:
        """Should raise ValueError if parquet is missing required columns."""
        parquet_path = tmp_path / "bad.parquet"
        pd.DataFrame([{"subject": "a", "rel": "b"}]).to_parquet(parquet_path)

        with pytest.raises(ValueError, match="missing columns"):
            list(stream_triples(preextracted_path=parquet_path))

    def test_raises_on_unsupported_extension(self, tmp_path: Path) -> None:
        """Should raise ValueError for unsupported file extensions."""
        csv_path = tmp_path / "triples.csv"
        csv_path.write_text("subject,relation,object\na,country,b\n")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            list(stream_triples(preextracted_path=csv_path))


# ---------------------------------------------------------------------------
# triple_extractor tests
# ---------------------------------------------------------------------------


class TestParseRebelOutput:
    """Tests for parse_rebel_output() — pure Python, no model needed."""

    def test_basic_triple(self) -> None:
        """Should parse a single well-formed REBEL output."""
        text = "<triplet> Albert Einstein <subj> Ulm <obj> place of birth"
        result = parse_rebel_output(text)
        assert len(result) == 1
        assert result[0]["subject"] == "albert einstein"
        assert result[0]["object"] == "ulm"
        assert result[0]["relation"] == "place_of_birth"

    def test_multiple_triples(self) -> None:
        """Should parse multiple <triplet> segments."""
        text = (
            "<triplet> Marie Curie <subj> Warsaw <obj> place of birth"
            "<triplet> Marie Curie <subj> Chemistry <obj> field of work"
        )
        result = parse_rebel_output(text)
        assert len(result) == 2

    def test_deduplication(self) -> None:
        """Duplicate triples within one output should be collapsed to one."""
        text = "<triplet> A <subj> B <obj> country" "<triplet> A <subj> B <obj> country"
        result = parse_rebel_output(text)
        assert len(result) == 1

    def test_empty_string(self) -> None:
        """Empty input should return an empty list."""
        assert parse_rebel_output("") == []

    def test_no_triplet_token(self) -> None:
        """Text without <triplet> should return empty list."""
        assert parse_rebel_output("Just some text without tokens.") == []

    def test_malformed_segment_skipped(self) -> None:
        """Segments missing <subj> or <obj> should be skipped gracefully."""
        text = "<triplet> incomplete segment without tokens"
        assert parse_rebel_output(text) == []

    def test_relation_normalised_to_snake_case(self) -> None:
        """Multi-word relation should become snake_case."""
        text = "<triplet> X <subj> Y <obj> place of birth"
        result = parse_rebel_output(text)
        assert result[0]["relation"] == "place_of_birth"


class TestFilterByRelation:
    """Tests for filter_by_relation()."""

    def test_keeps_allowed(self, sample_triple_dicts: list[Triple]) -> None:
        """All sample_triple_dicts use TOP_50_RELATIONS — none should be filtered."""
        result = filter_by_relation(sample_triple_dicts)
        assert len(result) == len(sample_triple_dicts)

    def test_removes_disallowed(self) -> None:
        """Triples with relations outside the allowed set should be dropped."""
        triples = [
            {"subject": "a", "relation": "not_in_set", "object": "b"},
            {"subject": "c", "relation": "country", "object": "d"},
        ]
        result = filter_by_relation(triples)
        assert len(result) == 1
        assert result[0]["relation"] == "country"

    def test_custom_allowed_set(self) -> None:
        """Should use the custom allowed set when provided."""
        triples = [
            {"subject": "a", "relation": "foo", "object": "b"},
            {"subject": "c", "relation": "bar", "object": "d"},
        ]
        result = filter_by_relation(triples, allowed=frozenset({"foo"}))
        assert len(result) == 1
        assert result[0]["relation"] == "foo"

    def test_does_not_mutate_input(self) -> None:
        """filter_by_relation should not modify the original list."""
        triples = [{"subject": "a", "relation": "country", "object": "b"}]
        original_len = len(triples)
        filter_by_relation(triples, allowed=frozenset())
        assert len(triples) == original_len


# ---------------------------------------------------------------------------
# embedder tests (slow — downloads model once, cached by sentence-transformers)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_embedder_produces_correct_shape() -> None:
    """embed_entities should return float32 array of shape (n, 384)."""
    from graphbench.ingestion.embedder import embed_entities

    result = embed_entities(["hello world", "albert einstein"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32


@pytest.mark.slow
def test_embedder_produces_unit_norm() -> None:
    """Each embedding should be L2-normalised to unit length."""
    from graphbench.ingestion.embedder import embed_entities

    result = embed_entities(["test entity"])
    norm = float(np.linalg.norm(result[0]))
    assert abs(norm - 1.0) < 1e-5


def test_embedder_raises_on_empty_list() -> None:
    """embed_entities should raise ValueError for empty input."""
    from graphbench.ingestion.embedder import embed_entities

    with pytest.raises(ValueError, match="must not be empty"):
        embed_entities([])


# ---------------------------------------------------------------------------
# faiss_writer tests
# ---------------------------------------------------------------------------


class TestFaissWriter:
    """Tests for build_and_save_index()."""

    def test_builds_searchable_index(
        self, tmp_path: Path, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Should build an index with ntotal equal to input length."""
        import faiss

        from graphbench.ingestion.faiss_writer import build_and_save_index

        entities, embeddings = tiny_embeddings
        index = build_and_save_index(
            entities, embeddings, index_path=tmp_path / "test_idx"
        )
        assert isinstance(index, faiss.IndexFlatIP)
        assert index.ntotal == len(entities)

    def test_saves_faiss_file(
        self, tmp_path: Path, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Should write a .faiss binary file to disk."""
        from graphbench.ingestion.faiss_writer import build_and_save_index

        entities, embeddings = tiny_embeddings
        build_and_save_index(entities, embeddings, index_path=tmp_path / "idx")
        assert (tmp_path / "idx.faiss").exists()

    def test_saves_id_map(
        self, tmp_path: Path, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Should write an id_map.json with correct entries."""
        from graphbench.ingestion.faiss_writer import build_and_save_index

        entities, embeddings = tiny_embeddings
        build_and_save_index(entities, embeddings, index_path=tmp_path / "idx")
        id_map_path = tmp_path / "idx_id_map.json"
        assert id_map_path.exists()
        id_map = json.loads(id_map_path.read_text())
        assert len(id_map) == len(entities)
        assert id_map["0"] == entities[0]

    def test_raises_on_shape_mismatch(self, tmp_path: Path) -> None:
        """Should raise ValueError if entity_strings and embeddings are mismatched."""
        from graphbench.ingestion.faiss_writer import build_and_save_index

        entities = ["a", "b"]
        embeddings = np.zeros((3, 384), dtype=np.float32)
        with pytest.raises(ValueError, match="entity_strings length"):
            build_and_save_index(entities, embeddings, index_path=tmp_path / "idx")

    def test_raises_on_wrong_dim(self, tmp_path: Path) -> None:
        """Should raise ValueError if embedding dimension is not 384."""
        from graphbench.ingestion.faiss_writer import build_and_save_index

        entities = ["a"]
        embeddings = np.zeros((1, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="384"):
            build_and_save_index(entities, embeddings, index_path=tmp_path / "idx")


# ---------------------------------------------------------------------------
# neo4j_writer tests (mocked)
# ---------------------------------------------------------------------------


class TestNeo4jWriter:
    """Tests for write_triples() using a mocked Neo4jClient."""

    def test_returns_triple_count(self, sample_triple_dicts: list[Triple]) -> None:
        """write_triples should return total number of triples written."""

        from graphbench.ingestion.neo4j_writer import write_triples

        mock_client = MagicMock()
        mock_client.execute_write.return_value = None

        result = write_triples(sample_triple_dicts, mock_client)
        assert result == len(sample_triple_dicts)

    def test_calls_execute_write_per_relation_type(
        self, sample_triple_dicts: list[Triple]
    ) -> None:
        """Should call execute_write for each batch, grouped by relation type."""

        from graphbench.ingestion.neo4j_writer import write_triples

        mock_client = MagicMock()
        mock_client.execute_write.return_value = None

        write_triples(sample_triple_dicts, mock_client)
        # 3 distinct relation types in sample_triple_dicts → at least 3 write calls
        assert mock_client.execute_write.call_count >= 3

    def test_idempotent_calls(self, sample_triple_dicts: list[Triple]) -> None:
        """Calling write_triples twice should use the same number of write calls."""

        from graphbench.ingestion.neo4j_writer import write_triples

        mock_client_1 = MagicMock()
        mock_client_2 = MagicMock()
        mock_client_1.execute_write.return_value = None
        mock_client_2.execute_write.return_value = None

        write_triples(sample_triple_dicts, mock_client_1)
        write_triples(sample_triple_dicts, mock_client_2)

        assert (
            mock_client_1.execute_write.call_count
            == mock_client_2.execute_write.call_count
        )

    def test_empty_input_returns_zero(self) -> None:
        """write_triples on empty list should return 0 without calling execute_write."""

        from graphbench.ingestion.neo4j_writer import write_triples

        mock_client = MagicMock()
        result = write_triples([], mock_client)
        assert result == 0
        mock_client.execute_write.assert_not_called()
