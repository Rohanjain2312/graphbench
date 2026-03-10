"""GraphRAG pipeline: community detection + LLM answer generation.

Pipeline A in the GraphBench head-to-head comparison.

Retrieval strategy:
1. Embed the question with all-MiniLM-L6-v2.
2. Top-10 FAISS search to find seed entities.
3. Extract 2-hop subgraph from Neo4j around seed entities.
4. Run Louvain community detection (resolution=0.8) on the subgraph.
5. Select the top-3 most relevant communities (by entity overlap with question).
6. Flatten community triples into a context string.
7. Pass context + question through PROMPT_TEMPLATE to Mistral-7B.

Must inherit from Pipeline and use build_prompt() for LLM formatting.

Implementation: Phase 4 (pipelines).
"""

from graphbench.pipelines.base import Pipeline, PipelineResult


class GraphRAGPipeline(Pipeline):
    """GraphRAG pipeline using Louvain community detection for context retrieval.

    Args:
        neo4j_client: Initialised Neo4jClient instance.
        faiss_client: Initialised FAISSClient instance.
        llm_client: Initialised LLMClient instance.
        community_detector: Initialised CommunityDetector instance.
    """

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return "GraphRAG"

    def answer(self, question: str) -> PipelineResult:
        """Answer a question using community-based context retrieval.

        Args:
            question: Natural language question string.

        Returns:
            PipelineResult with predicted_answer and context_triples.
        """
        # TODO: Phase 4 implementation
        raise NotImplementedError("GraphRAGPipeline.answer() — implement in Phase 4")
