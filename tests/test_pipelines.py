"""Tests for graphbench.pipelines.

Covers Pipeline ABC, PipelineResult, GraphRAGPipeline, GNNRAGPipeline.
Full pipeline tests in Phase 4.
"""

import pytest

from graphbench.pipelines.base import PROMPT_TEMPLATE, Pipeline, PipelineResult
from graphbench.pipelines.gnnrag_pipeline import GNNRAGPipeline
from graphbench.pipelines.graphrag_pipeline import GraphRAGPipeline


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


class TestGraphRAGPipeline:
    """Structural tests for GraphRAGPipeline stub."""

    def test_name(self) -> None:
        p = GraphRAGPipeline()
        assert p.name == "GraphRAG"

    @pytest.mark.skip(reason="Phase 4 — not yet implemented")
    def test_answer_returns_pipeline_result(self) -> None:
        pass


class TestGNNRAGPipeline:
    """Structural tests for GNNRAGPipeline stub."""

    def test_name(self) -> None:
        p = GNNRAGPipeline()
        assert p.name == "GNN-RAG"

    @pytest.mark.skip(reason="Phase 4 — not yet implemented")
    def test_answer_returns_pipeline_result(self) -> None:
        pass
