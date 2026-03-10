"""Tests for graphbench.pipelines and graphbench.community.

Covers:
- Pipeline ABC, PipelineResult, PROMPT_TEMPLATE (Phase 1)
- GraphRAGPipeline.answer() with mocked dependencies (Phase 4)
- GNNRAGPipeline.answer() with mocked dependencies (Phase 4)
- CommunityDetector (Phase 4)
- merge_community_triples / _rank_by_entity_frequency (Phase 4)
- LLMClient backend resolution (Phase 4)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graphbench.community.detector import CommunityDetector
from graphbench.community.summarizer import merge_community_triples
from graphbench.pipelines.base import PROMPT_TEMPLATE, Pipeline, PipelineResult
from graphbench.pipelines.gnnrag_pipeline import GNNRAGPipeline
from graphbench.pipelines.graphrag_pipeline import GraphRAGPipeline

# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for the PipelineResult dataclass."""

    def test_default_values(self) -> None:
        result = PipelineResult(question="test?", predicted_answer="answer")
        assert result.gold_answer is None
        assert result.context_triples == []
        assert result.latency_ms == 0.0
        assert result.pipeline_name == ""
        assert result.metadata == {}

    def test_with_all_fields(self) -> None:
        result = PipelineResult(
            question="Where was Einstein born?",
            predicted_answer="Ulm",
            gold_answer="Ulm",
            context_triples=[("Einstein", "born_in", "Ulm")],
            latency_ms=42.5,
            pipeline_name="GraphRAG",
        )
        assert result.predicted_answer == "Ulm"
        assert len(result.context_triples) == 1


# ---------------------------------------------------------------------------
# PROMPT_TEMPLATE and build_prompt
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    """Tests for the shared PROMPT_TEMPLATE."""

    def test_template_contains_placeholders(self) -> None:
        assert "{context}" in PROMPT_TEMPLATE
        assert "{question}" in PROMPT_TEMPLATE

    def test_build_prompt_formats_correctly(self, sample_triples) -> None:
        """build_prompt() should insert context and question into the template."""

        class ConcretePipeline(Pipeline):
            @property
            def name(self) -> str:
                return "Test"

            def answer(self, question: str) -> PipelineResult:
                raise NotImplementedError

        pipeline = ConcretePipeline()
        prompt = pipeline.build_prompt("Where was Einstein born?", sample_triples[:2])
        assert "Where was Einstein born?" in prompt
        assert "Albert Einstein" in prompt

    def test_build_prompt_no_triples(self) -> None:
        """build_prompt() with empty triples should include 'No context available.'."""

        class ConcretePipeline(Pipeline):
            @property
            def name(self) -> str:
                return "Test"

            def answer(self, question: str) -> PipelineResult:
                raise NotImplementedError

        pipeline = ConcretePipeline()
        prompt = pipeline.build_prompt("What?", [])
        assert "No context available." in prompt


# ---------------------------------------------------------------------------
# GraphRAGPipeline
# ---------------------------------------------------------------------------


class TestGraphRAGPipeline:
    """Tests for GraphRAGPipeline."""

    def test_name(self) -> None:
        p = GraphRAGPipeline()
        assert p.name == "GraphRAG"

    def test_answer_raises_without_clients(self) -> None:
        """answer() should raise RuntimeError if required clients are not set."""
        p = GraphRAGPipeline()
        with pytest.raises(RuntimeError, match="neo4j_client|faiss_client|llm_client"):
            p.answer("Who was Einstein?")

    def _make_pipeline(self, triples, llm_answer="Ulm"):
        """Helper: build a GraphRAGPipeline with all deps mocked."""
        mock_neo4j = MagicMock()
        mock_neo4j.get_subgraph_multi.return_value = triples

        mock_faiss = MagicMock()
        mock_faiss.search.return_value = [
            ("Albert Einstein", 0.95),
            ("Ulm", 0.80),
        ]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_answer

        # Patch sentence transformer embed to avoid model download
        with patch.object(
            GraphRAGPipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            pipeline = GraphRAGPipeline(
                neo4j_client=mock_neo4j,
                faiss_client=mock_faiss,
                llm_client=mock_llm,
            )
        return pipeline, mock_neo4j, mock_faiss, mock_llm

    def test_answer_returns_pipeline_result(self, sample_triples) -> None:
        """answer() should return a PipelineResult instance."""
        pipeline, _, _, _ = self._make_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert isinstance(result, PipelineResult)

    def test_answer_sets_pipeline_name(self, sample_triples) -> None:
        """answer() should set pipeline_name to 'GraphRAG'."""
        pipeline, _, _, _ = self._make_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert result.pipeline_name == "GraphRAG"

    def test_answer_has_context_triples(self, sample_triples) -> None:
        """answer() should populate context_triples from the subgraph."""
        pipeline, _, _, _ = self._make_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert len(result.context_triples) > 0

    def test_answer_predicted_answer_from_llm(self, sample_triples) -> None:
        """answer() should return whatever the LLM generates."""
        pipeline, _, _, _ = self._make_pipeline(sample_triples, llm_answer="Ulm")

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert result.predicted_answer == "Ulm"

    def test_answer_empty_subgraph_returns_i_dont_know(self) -> None:
        """answer() with empty subgraph should return 'I don't know.' gracefully."""
        pipeline, mock_neo4j, _, _ = self._make_pipeline([])
        mock_neo4j.get_subgraph_multi.return_value = []

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Unknown question?")

        assert "don't know" in result.predicted_answer.lower()
        assert result.context_triples == []

    def test_answer_empty_faiss_returns_i_dont_know(self) -> None:
        """answer() with no FAISS results should return 'I don't know.' gracefully."""
        pipeline, _, mock_faiss, _ = self._make_pipeline([])
        mock_faiss.search.return_value = []

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Unknown question?")

        assert "don't know" in result.predicted_answer.lower()

    def test_answer_metadata_contains_seed_entities(self, sample_triples) -> None:
        """answer() metadata should include seed_entities list."""
        pipeline, _, _, _ = self._make_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert "seed_entities" in result.metadata
        assert isinstance(result.metadata["seed_entities"], list)


# ---------------------------------------------------------------------------
# GNNRAGPipeline
# ---------------------------------------------------------------------------


class TestGNNRAGPipeline:
    """Tests for GNNRAGPipeline."""

    def test_name(self) -> None:
        p = GNNRAGPipeline()
        assert p.name == "GNN-RAG"

    def test_answer_raises_without_clients(self) -> None:
        """answer() should raise RuntimeError if required clients are not set."""
        p = GNNRAGPipeline()
        with pytest.raises(
            RuntimeError, match="neo4j_client|faiss_client|llm_client|gat_model"
        ):
            p.answer("Who was Einstein?")

    def _make_gnn_pipeline(self, triples, llm_answer="Ulm"):
        """Helper: build a GNNRAGPipeline with all deps mocked."""
        import torch  # lazy

        mock_neo4j = MagicMock()
        mock_neo4j.get_subgraph_multi.return_value = triples

        mock_faiss = MagicMock()
        mock_faiss.search.return_value = [
            ("Albert Einstein", 0.95),
            ("Ulm", 0.80),
        ]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_answer

        n_edges = len(triples) if triples else 1
        mock_model = MagicMock()
        mock_model.score_edges.return_value = torch.ones(n_edges)

        # Simple entity embeddings dict
        rng = np.random.default_rng(0)
        entities_in_triples = set()
        for s, _, o in triples:
            entities_in_triples.add(s)
            entities_in_triples.add(o)
        entity_embeddings = {
            e: rng.standard_normal(384).astype(np.float32) for e in entities_in_triples
        }

        pipeline = GNNRAGPipeline(
            neo4j_client=mock_neo4j,
            faiss_client=mock_faiss,
            llm_client=mock_llm,
            gat_model=mock_model,
            entity_embeddings=entity_embeddings,
        )
        return pipeline, mock_neo4j, mock_faiss, mock_llm, mock_model

    @pytest.mark.gnn
    def test_answer_returns_pipeline_result(self, sample_triples) -> None:
        """answer() should return a PipelineResult instance."""
        pipeline, *_ = self._make_gnn_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert isinstance(result, PipelineResult)

    @pytest.mark.gnn
    def test_answer_sets_pipeline_name(self, sample_triples) -> None:
        """answer() should set pipeline_name to 'GNN-RAG'."""
        pipeline, *_ = self._make_gnn_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert result.pipeline_name == "GNN-RAG"

    @pytest.mark.gnn
    def test_answer_has_context_triples(self, sample_triples) -> None:
        """answer() should populate context_triples from top-scored edges."""
        pipeline, *_ = self._make_gnn_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert len(result.context_triples) > 0

    @pytest.mark.gnn
    def test_answer_predicted_answer_from_llm(self, sample_triples) -> None:
        """answer() should return whatever the LLM generates."""
        pipeline, *_ = self._make_gnn_pipeline(sample_triples, llm_answer="Ulm")

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert result.predicted_answer == "Ulm"

    @pytest.mark.gnn
    def test_answer_empty_subgraph_returns_i_dont_know(self) -> None:
        """answer() with empty subgraph should return 'I don't know.' gracefully."""
        pipeline, mock_neo4j, _, _, _ = self._make_gnn_pipeline([])
        mock_neo4j.get_subgraph_multi.return_value = []

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Unknown question?")

        assert "don't know" in result.predicted_answer.lower()

    @pytest.mark.gnn
    def test_answer_metadata_contains_seed_entities(self, sample_triples) -> None:
        """answer() metadata should include seed_entities."""
        pipeline, *_ = self._make_gnn_pipeline(sample_triples)

        with patch.object(
            pipeline,
            "_embed_question",
            return_value=np.zeros(384, dtype=np.float32),
        ):
            result = pipeline.answer("Where was Einstein born?")

        assert "seed_entities" in result.metadata


# ---------------------------------------------------------------------------
# CommunityDetector
# ---------------------------------------------------------------------------


class TestCommunityDetector:
    """Tests for CommunityDetector."""

    @pytest.fixture
    def simple_graph(self):
        """A small undirected NetworkX graph with clear community structure."""
        import networkx as nx

        graph = nx.Graph()
        # Community 1: a-b-c tightly connected
        graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        # Community 2: d-e-f tightly connected
        graph.add_edges_from([("d", "e"), ("e", "f"), ("d", "f")])
        # Weak bridge
        graph.add_edge("c", "d")
        return graph

    def test_detect_returns_partition_dict(self, simple_graph) -> None:
        """detect() should return a dict mapping node → community ID."""
        detector = CommunityDetector()
        partition = detector.detect(simple_graph)
        assert isinstance(partition, dict)
        assert set(partition.keys()) == {"a", "b", "c", "d", "e", "f"}

    def test_detect_empty_graph_returns_empty(self) -> None:
        """detect() on an empty graph should return {}."""
        import networkx as nx

        detector = CommunityDetector()
        assert detector.detect(nx.Graph()) == {}

    def test_detect_digraph_auto_undirected(self) -> None:
        """detect() on a DiGraph should convert to undirected without error."""
        import networkx as nx

        digraph = nx.DiGraph()
        digraph.add_edges_from([("x", "y"), ("y", "z"), ("z", "x")])
        detector = CommunityDetector()
        partition = detector.detect(digraph)
        assert set(partition.keys()) == {"x", "y", "z"}

    def test_group_triples_by_community(self, sample_triples) -> None:
        """group_triples() should group triples by subject's community ID."""
        detector = CommunityDetector()
        partition = {"Albert Einstein": 0, "Ulm": 0, "Marie Curie": 1, "Warsaw": 1}
        groups = detector.group_triples(partition, sample_triples)
        assert isinstance(groups, dict)
        # Every triple appears in exactly one group
        all_triples = [t for ts in groups.values() for t in ts]
        assert len(all_triples) == len(sample_triples)

    def test_select_top_communities_by_overlap(self, sample_triples) -> None:
        """select_top_communities() should prefer communities with seed overlap."""
        detector = CommunityDetector()
        groups = {
            0: [("Albert Einstein", "born_in", "Ulm")],
            1: [("Marie Curie", "born_in", "Warsaw")],
        }
        selected = detector.select_top_communities(
            groups, seed_entities=["Albert Einstein"], k=1
        )
        assert len(selected) == 1
        assert selected[0] == 0  # community with Einstein should win

    def test_select_top_communities_empty_groups(self) -> None:
        """select_top_communities() with empty groups should return []."""
        detector = CommunityDetector()
        assert detector.select_top_communities({}, seed_entities=["x"], k=3) == []


# ---------------------------------------------------------------------------
# merge_community_triples
# ---------------------------------------------------------------------------


class TestMergeCommunityTriples:
    """Tests for community.summarizer.merge_community_triples."""

    @pytest.fixture
    def groups(self) -> dict:
        return {
            0: [
                ("Alice", "knows", "Bob"),
                ("Alice", "knows", "Carol"),
                ("Bob", "knows", "Alice"),
            ],
            1: [
                ("Dave", "works_at", "Acme"),
                ("Eve", "works_at", "Acme"),
            ],
        }

    def test_merges_selected_communities(self, groups) -> None:
        result = merge_community_triples(groups, selected_ids=[0, 1])
        assert len(result) == 5

    def test_respects_max_triples(self, groups) -> None:
        result = merge_community_triples(groups, selected_ids=[0, 1], max_triples=2)
        assert len(result) == 2

    def test_ranked_puts_central_entities_first(self, groups) -> None:
        """With ranked=True, most-connected entities should appear first."""
        result = merge_community_triples(groups, selected_ids=[0], ranked=True)
        # Alice appears in 3 triples — should be subject of first triple
        assert result[0][0] == "Alice" or result[0][2] == "Alice"

    def test_empty_selected_ids(self, groups) -> None:
        result = merge_community_triples(groups, selected_ids=[])
        assert result == []

    def test_missing_community_id_skipped(self, groups) -> None:
        result = merge_community_triples(groups, selected_ids=[99])
        assert result == []


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class TestLLMClientInit:
    """Tests for LLMClient backend resolution (no actual model loading)."""

    def test_ollama_backend_resolves_when_reachable(self) -> None:
        """LLMClient with backend='ollama' should use ollama backend."""
        from graphbench.utils.llm_client import LLMClient

        with patch(
            "graphbench.utils.llm_client.LLMClient._ollama_reachable",
            return_value=True,
        ):
            client = LLMClient(backend="ollama", model="phi3")
        assert client.backend == "ollama"
        assert client.model == "phi3"

    def test_hf_backend_forced(self) -> None:
        """LLMClient with backend='hf' should use HF backend."""
        from graphbench.utils.llm_client import LLMClient

        client = LLMClient(backend="hf", model="mistralai/Mistral-7B-Instruct-v0.2")
        assert client.backend == "hf"

    def test_auto_falls_back_to_hf_when_ollama_unreachable(self) -> None:
        """backend='auto' should fall back to HF when Ollama is down."""
        from graphbench.utils.llm_client import LLMClient

        with patch(
            "graphbench.utils.llm_client.LLMClient._ollama_reachable",
            return_value=False,
        ):
            client = LLMClient(backend="auto")
        assert client.backend == "hf"

    def test_auto_prefers_ollama_when_reachable(self) -> None:
        """backend='auto' should select Ollama when it is reachable."""
        from graphbench.utils.llm_client import LLMClient

        with patch(
            "graphbench.utils.llm_client.LLMClient._ollama_reachable",
            return_value=True,
        ):
            client = LLMClient(backend="auto")
        assert client.backend == "ollama"

    def test_generate_ollama_calls_requests(self) -> None:
        """generate() with Ollama backend should POST to the Ollama API."""
        from graphbench.utils.llm_client import LLMClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Berlin"}

        with patch(
            "graphbench.utils.llm_client.LLMClient._ollama_reachable", return_value=True
        ):
            client = LLMClient(backend="ollama", model="phi3")

        with patch("requests.post", return_value=mock_resp) as mock_post:
            answer = client.generate("Where was Einstein born?")

        mock_post.assert_called_once()
        assert answer == "Berlin"
