"""GNN-RAG pipeline: GAT-scored subgraph traversal + LLM answer generation.

Pipeline B in the GraphBench head-to-head comparison.

Retrieval strategy:
1. Embed the question with all-MiniLM-L6-v2.
2. Top-K FAISS search to find seed entities.
3. Extract 2-hop subgraph from Neo4j around seed entities.
4. Convert subgraph to PyG Data using pre-loaded entity embeddings.
5. Score all edges with the trained 3-layer GAT model.
6. Retain the top-``top_edges`` edges by GAT score.
7. Pass context + question through PROMPT_TEMPLATE to the LLM.

GAT model must have AUC-ROC > 0.75 on the held-out test set before use
in a benchmark run (enforced by the trainer gate in Phase 3).

Depends on:
- :class:`~graphbench.utils.neo4j_client.Neo4jClient`
- :class:`~graphbench.utils.faiss_client.FAISSClient`
- :class:`~graphbench.utils.llm_client.LLMClient`
- :class:`~graphbench.gnn.model.GATModel` (trained checkpoint)
- ``entity_embeddings``: ``dict[str, np.ndarray]`` of pre-built node features
"""

from __future__ import annotations

import logging

import numpy as np

from graphbench.pipelines.base import Pipeline, PipelineResult
from graphbench.utils.config import settings

logger = logging.getLogger(__name__)


class GNNRAGPipeline(Pipeline):
    """GNN-RAG pipeline using a 3-layer GAT for edge scoring and context selection.

    Args:
        neo4j_client: Initialised :class:`~graphbench.utils.neo4j_client.Neo4jClient`.
        faiss_client: Initialised :class:`~graphbench.utils.faiss_client.FAISSClient`.
        llm_client: Initialised :class:`~graphbench.utils.llm_client.LLMClient`.
        gat_model: Trained :class:`~graphbench.gnn.model.GATModel` loaded from checkpoint.
        entity_embeddings: Pre-built entity embedding lookup ``{entity_string: np.ndarray}``.
            Entities not in this dict receive zero-vector node features.
        top_edges: Number of top-scoring edges to include in context (default: 50).
    """

    def __init__(
        self,
        neo4j_client=None,
        faiss_client=None,
        llm_client=None,
        gat_model=None,
        entity_embeddings: dict[str, np.ndarray] | None = None,
        top_edges: int = 50,
    ) -> None:
        self._neo4j = neo4j_client
        self._faiss = faiss_client
        self._llm = llm_client
        self._model = gat_model
        self._entity_embeddings: dict[str, np.ndarray] = entity_embeddings or {}
        self._top_edges = top_edges
        self._embedder = None  # lazy-loaded once on first call

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return "GNN-RAG"

    def answer(self, question: str) -> PipelineResult:
        """Answer a question using GAT-scored subgraph context retrieval.

        Steps:
        1. Embed question → FAISS top-K seed entities.
        2. 2-hop Neo4j subgraph from seed entities.
        3. Convert to PyG Data with pre-built entity embeddings.
        4. Score all edges with trained GATModel.
        5. Select top-N edges by score.
        6. Generate answer with LLM.

        Args:
            question: Natural language question string.

        Returns:
            PipelineResult with ``predicted_answer``, ``context_triples``,
            and ``metadata`` containing edge scores.

        Raises:
            RuntimeError: If required clients (neo4j, faiss, llm, gat_model) are not set.
        """
        self._check_clients(
            [
                ("neo4j_client", self._neo4j),
                ("faiss_client", self._faiss),
                ("llm_client", self._llm),
                ("gat_model", self._model),
            ]
        )

        # ── Step 1: embed question → seed entities ──────────────────────
        query_vec = self._embed_question(question)
        faiss_results = self._faiss.search(query_vec, k=settings.top_k_faiss)
        seed_entities = [entity for entity, _ in faiss_results]
        logger.debug("GNN-RAG: %d seed entities from FAISS.", len(seed_entities))

        if not seed_entities:
            return self._empty_result(question, "No seed entities found via FAISS.")

        # ── Step 2: extract 2-hop subgraph from Neo4j ───────────────────
        triples = self._neo4j.get_subgraph_multi(
            seed_entities, hops=settings.subgraph_hops
        )
        logger.debug("GNN-RAG: %d triples in subgraph.", len(triples))

        if not triples:
            return self._empty_result(question, "Empty subgraph from Neo4j.")

        # ── Step 3: convert subgraph to PyG Data ─────────────────────────
        try:
            from graphbench.gnn.subgraph import subgraph_to_pyg  # noqa: PLC0415

            data = subgraph_to_pyg(triples, self._entity_embeddings)
        except ValueError as exc:
            return self._empty_result(question, str(exc))

        # ── Step 4: score all edges with GAT ────────────────────────────
        import torch  # noqa: PLC0415

        # Move tensors to the same device as the model weights
        model_device = next(self._model.parameters()).device
        x = data.x.to(model_device)
        edge_index = data.edge_index.to(model_device)

        self._model.eval()
        with torch.no_grad():
            scores = self._model.score_edges(x, edge_index, edge_index)

        # ── Step 5: select top-N edges by score ─────────────────────────
        n_edges = data.edge_index.shape[1]
        k = min(self._top_edges, n_edges)
        top_indices = torch.topk(scores, k=k).indices.tolist()

        # Reconstruct original triples for selected edge indices
        context_triples = self._indices_to_triples(top_indices, data, triples)

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
                "n_subgraph_triples": len(triples),
                "n_scored_edges": n_edges,
                "top_edge_scores": [float(scores[i]) for i in top_indices],
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

    def _indices_to_triples(
        self,
        edge_indices: list[int],
        data,
        original_triples: list[tuple[str, str, str]],
    ) -> list[tuple[str, str, str]]:
        """Map PyG edge indices back to original (subject, relation, object) triples.

        The PyG ``edge_index`` preserves insertion order from ``subgraph_to_pyg``.
        We use ``data.node_index`` to reverse-map integer node IDs to entity strings,
        then look up the corresponding triple by (src_entity, dst_entity) pair.

        Args:
            edge_indices: List of edge indices in ``data.edge_index`` to retrieve.
            data: PyG Data object with ``edge_index`` and ``node_index`` attributes.
            original_triples: Original ``(subj, rel, obj)`` list for relation lookup.

        Returns:
            List of ``(subject, relation, object)`` triples for the selected edges.
        """
        # Reverse node_index: int → entity string
        inv_node = {v: k for k, v in data.node_index.items()}

        # Build a lookup for relation: (subj_lower, obj_lower) → (subj, rel, obj)
        rel_lookup: dict[tuple[str, str], tuple[str, str, str]] = {}
        for subj, rel, obj in original_triples:
            rel_lookup[(subj.lower(), obj.lower())] = (subj, rel, obj)

        result: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        edge_index = data.edge_index
        for idx in edge_indices:
            src_id = int(edge_index[0, idx])
            dst_id = int(edge_index[1, idx])
            src_entity = inv_node.get(src_id, "")
            dst_entity = inv_node.get(dst_id, "")
            triple = rel_lookup.get(
                (src_entity.lower(), dst_entity.lower()),
                (src_entity, "related_to", dst_entity),
            )
            if triple not in seen:
                seen.add(triple)
                result.append(triple)

        return result
