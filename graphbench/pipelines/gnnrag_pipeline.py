"""GNN-RAG pipeline: GAT-scored subgraph traversal + LLM answer generation.

Pipeline B in the GraphBench head-to-head comparison.

Retrieval strategy:
1. Embed the question with all-MiniLM-L6-v2.
2. Top-10 FAISS search to find seed entities.
3. Extract 2-hop subgraph from Neo4j around seed entities.
4. Score all edges in the subgraph using the trained 3-layer GAT model.
5. Retain top-50 edges by GAT attention score.
6. Flatten scored triples into a context string.
7. Pass context + question through PROMPT_TEMPLATE to Mistral-7B.

Must inherit from Pipeline and use build_prompt() for LLM formatting.
GAT model must have AUC-ROC > 0.75 on held-out test set before use.

Implementation: Phase 4 (pipelines).
"""

from graphbench.pipelines.base import Pipeline, PipelineResult


class GNNRAGPipeline(Pipeline):
    """GNN-RAG pipeline using a 3-layer GAT for edge scoring and context selection.

    Args:
        neo4j_client: Initialised Neo4jClient instance.
        faiss_client: Initialised FAISSClient instance.
        llm_client: Initialised LLMClient instance.
        gat_model: Trained GATModel loaded from checkpoint.
    """

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return "GNN-RAG"

    def answer(self, question: str) -> PipelineResult:
        """Answer a question using GAT-scored subgraph context retrieval.

        Args:
            question: Natural language question string.

        Returns:
            PipelineResult with predicted_answer and context_triples.
        """
        # TODO: Phase 4 implementation
        raise NotImplementedError("GNNRAGPipeline.answer() — implement in Phase 4")
