"""Tests for graphbench.utils.faiss_client."""

from pathlib import Path

import numpy as np
import pytest

from graphbench.utils.faiss_client import FAISSClient


class TestFAISSClientBuild:
    """Tests for FAISSClient.build() (in-memory construction)."""

    def test_size_matches_input(
        self, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """size property should equal number of input entities."""
        entities, embeddings = tiny_embeddings
        client = FAISSClient.build(entities, embeddings)
        assert client.size == len(entities)

    def test_top_k_search_returns_k_results(
        self, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """search() should return exactly k results when k < index size."""
        entities, embeddings = tiny_embeddings
        client = FAISSClient.build(entities, embeddings)
        results = client.search(embeddings[0], k=3)
        assert len(results) == 3

    def test_top_1_search_finds_self(
        self, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Searching with a vector already in the index should find itself first."""
        entities, embeddings = tiny_embeddings
        client = FAISSClient.build(entities, embeddings)
        results = client.search(embeddings[0], k=1)
        assert results[0][0] == entities[0]

    def test_self_similarity_is_near_one(
        self, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Cosine similarity of a unit vector with itself should be ~1.0."""
        entities, embeddings = tiny_embeddings
        client = FAISSClient.build(entities, embeddings)
        results = client.search(embeddings[2], k=1)
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_results_sorted_descending(
        self, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Results should be sorted by score in descending order."""
        entities, embeddings = tiny_embeddings
        client = FAISSClient.build(entities, embeddings)
        results = client.search(embeddings[0], k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_handles_1d_and_2d_query(
        self, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """search() should accept both (384,) and (1, 384) shaped queries."""
        entities, embeddings = tiny_embeddings
        client = FAISSClient.build(entities, embeddings)
        r1 = client.search(embeddings[0], k=1)
        r2 = client.search(embeddings[0].reshape(1, -1), k=1)
        assert r1[0][0] == r2[0][0]


class TestFAISSClientLoadSave:
    """Tests for FAISSClient.load() and the faiss_writer integration."""

    def test_save_and_load_round_trip(
        self, tmp_path: Path, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Loaded client should return identical search results as in-memory build."""
        from graphbench.ingestion.faiss_writer import build_and_save_index

        entities, embeddings = tiny_embeddings
        build_and_save_index(entities, embeddings, index_path=tmp_path / "idx")
        loaded = FAISSClient.load(tmp_path / "idx")

        built = FAISSClient.build(entities, embeddings)
        loaded_results = loaded.search(embeddings[0], k=3)
        built_results = built.search(embeddings[0], k=3)

        assert [r[0] for r in loaded_results] == [r[0] for r in built_results]

    def test_load_raises_if_faiss_missing(self, tmp_path: Path) -> None:
        """FAISSClient.load() should raise FileNotFoundError if .faiss is absent."""
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            FAISSClient.load(tmp_path / "nonexistent")

    def test_load_raises_if_id_map_missing(
        self, tmp_path: Path, tiny_embeddings: tuple[list[str], np.ndarray]
    ) -> None:
        """Should raise FileNotFoundError if id_map.json is absent."""
        import faiss

        # Write only the .faiss file, no id_map
        entities, embeddings = tiny_embeddings
        idx = faiss.IndexFlatIP(384)
        idx.add(embeddings)
        faiss.write_index(idx, str(tmp_path / "idx.faiss"))

        with pytest.raises(FileNotFoundError, match="id_map not found"):
            FAISSClient.load(tmp_path / "idx")
