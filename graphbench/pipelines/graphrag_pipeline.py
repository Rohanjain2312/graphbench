"""GraphRAG pipeline: community detection + LLM answer generation.

Pipeline A in the GraphBench head-to-head comparison.

Retrieval strategy:
1. Embed the question with all-MiniLM-L6-v2.
2. Top-K FAISS search to find seed entities.
3. Extract 2-hop subgraph from Neo4j around seed entities.
4. Run Louvain community detection (resolution=0.8) on the subgraph.
5. Select the top-3 most relevant communities (by entity overlap with seed entities).
6. Merge and rank community triples by entity centrality.
7. Pass context + question through PROMPT_TEMPLATE to the LLM.

Depends on:
- :class:`~graphbench.utils.neo4j_client.Neo4jClient`
- :class:`~graphbench.utils.faiss_client.FAISSClient`
- :class:`~graphbench.utils.llm_client.LLMClient`
- :class:`~graphbench.community.detector.CommunityDetector`
"""

from __future__ import annotations

import logging

import numpy as np

from graphbench.community.detector import CommunityDetector
from graphbench.community.summarizer import merge_community_triples
from graphbench.pipelines.base import Pipeline, PipelineResult
from graphbench.utils.config import settings

logger = logging.getLogger(__name__)


class GraphRAGPipeline(Pipeline):
    """GraphRAG pipeline using Louvain community detection for context retrieval.

    Args:
        neo4j_client: Initialised :class:`~graphbench.utils.neo4j_client.Neo4jClient`.
        faiss_client: Initialised :class:`~graphbench.utils.faiss_client.FAISSClient`.
        llm_client: Initialised :class:`~graphbench.utils.llm_client.LLMClient`.
        community_detector: Initialised :class:`~graphbench.community.detector.CommunityDetector`.
            Created with default settings if not provided.
        top_communities: Number of top communities to include in context (default: 3).
        max_context_triples: Hard cap on context triples passed to LLM (default: 100).
    """

    def __init__(
        self,
        neo4j_client=None,
        faiss_client=None,
        llm_client=None,
        community_detector: CommunityDetector | None = None,
        top_communities: int = 3,
        max_context_triples: int = 100,
    ) -> None:
        self._neo4j = neo4j_client
        self._faiss = faiss_client
        self._llm = llm_client
        self._detector = community_detector or CommunityDetector()
        self._top_communities = top_communities
        self._max_context_triples = max_context_triples
        self._embedder = None  # lazy-loaded once on first call

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return "GraphRAG"

    def answer(self, question: str) -> PipelineResult:
        """Answer a question using community-based context retrieval.

        Steps:
        1. Embed question → FAISS top-K seed entities.
        2. 2-hop Neo4j subgraph from seed entities.
        3. Louvain community detection on subgraph.
        4. Select top-K communities by seed-entity overlap.
        5. Merge and rank community triples.
        6. Generate answer with LLM.

        Args:
            question: Natural language question string.

        Returns:
            PipelineResult with ``predicted_answer``, ``context_triples``,
            and ``metadata`` containing community IDs.

        Raises:
            RuntimeError: If required clients (neo4j, faiss, llm) are not set.
        """
        self._check_clients(
            [
                ("neo4j_client", self._neo4j),
                ("faiss_client", self._faiss),
                ("llm_client", self._llm),
            ]
        )

        # ── Step 1: embed question → seed entities ──────────────────────
        query_vec = self._embed_question(question)
        faiss_results = self._faiss.search(query_vec, k=settings.top_k_faiss)
        seed_entities = [entity for entity, _ in faiss_results]
        logger.debug("GraphRAG: %d seed entities from FAISS.", len(seed_entities))

        if not seed_entities:
            return self._empty_result(question, "No seed entities found via FAISS.")

        # ── Step 2: extract 2-hop subgraph from Neo4j ───────────────────
        triples = self._neo4j.get_subgraph_multi(
            seed_entities, hops=settings.subgraph_hops
        )
        logger.debug("GraphRAG: %d triples in subgraph.", len(triples))

        if not triples:
            return self._empty_result(question, "Empty subgraph from Neo4j.")

        # ── Step 3: Louvain community detection ─────────────────────────
        from graphbench.gnn.subgraph import subgraph_to_networkx  # noqa: PLC0415

        graph = subgraph_to_networkx(triples)
        partition = self._detector.detect(graph)
        groups = self._detector.group_triples(partition, triples)

        # ── Step 4: select top communities ──────────────────────────────
        selected_ids = self._detector.select_top_communities(
            groups, seed_entities, k=self._top_communities
        )

        # ── Step 5: merge + rank community triples ───────────────────────
        context_triples = merge_community_triples(
            groups,
            selected_ids,
            max_triples=self._max_context_triples,
        )

        # ── Step 6: LLM generation ───────────────────────────────────────
        prompt = self.build_prompt(question, context_triples)
        predicted_answer = self._llm.generate(prompt)

        return PipelineResult(
            question=question,
            predicted_answer=predicted_answer,
            context_triples=context_triples,
            pipeline_name=self.name,
            metadata={
                "seed_entities": seed_entities,
                "community_ids": selected_ids,
                "n_communities_total": len(groups),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_question(self, question: str) -> np.ndarray:
        """Embed question with SentenceTransformer (lazy import, cached).

        The SentenceTransformer model is loaded once on the first call and
        reused for all subsequent questions, avoiding the 500× reload overhead
        during a full benchmark run.

        Args:
            question: Question string to embed.

        Returns:
            L2-normalised float32 embedding of shape ``(384,)``.
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._embedder = SentenceTransformer(settings.embedding_model)
        vec = self._embedder.encode([question], normalize_embeddings=True)
        return vec[0].astype(np.float32)
