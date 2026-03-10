"""Subgraph extraction and PyG conversion utilities.

Given a list of (subject, relation, object) triples extracted from Neo4j,
converts them into a PyG Data object ready for GAT forward pass.

Used by both pipelines at inference time:
- GraphRAGPipeline: feeds the Data to Louvain community detection via NetworkX.
- GNNRAGPipeline: feeds the Data to GATModel.encode() + .decode().

Entities absent from the embedding index receive zero-vector features.
Subgraphs exceeding max_nodes are trimmed by entity degree (highest-degree kept).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384


def subgraph_to_pyg(
    triples: list[tuple[str, str, str]],
    entity_embeddings: dict[str, np.ndarray],
    *,
    embedding_dim: int = _EMBEDDING_DIM,
    max_nodes: int = 500,
) -> Data:
    """Convert a Neo4j subgraph to a PyG Data object for GAT inference.

    Args:
        triples: List of (subject, relation, object) tuples from
            Neo4jClient.get_subgraph() or get_subgraph_multi().
        entity_embeddings: Mapping from entity surface string to embedding array.
            Entities absent from this dict receive zero-vector node features.
        embedding_dim: Node feature dimensionality (default 384).
        max_nodes: If the subgraph exceeds this many nodes, trim to the
            highest-degree entities. 500 fits comfortably in GAT memory on Colab.

    Returns:
        PyG Data with:
        - x (Tensor[N, embedding_dim]): Node feature matrix.
        - edge_index (Tensor[2, E]): Directed edge connectivity.
        - node_index (dict[str, int]): Entity string → node ID mapping.
        - entities (list[str]): Node strings, index-aligned with x rows.
        - num_nodes (int): Total node count.

    Raises:
        ValueError: If triples is empty.
    """
    if not triples:
        raise ValueError("triples must not be empty to build a PyG subgraph.")

    # Optionally trim to max_nodes by entity degree
    if _count_unique_entities(triples) > max_nodes:
        triples = _trim_by_degree(triples, max_nodes)
        logger.debug(
            "Subgraph trimmed to %d nodes (max_nodes=%d).", max_nodes, max_nodes
        )

    # Build sorted entity list and index map
    entities = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
    node_index: dict[str, int] = {e: i for i, e in enumerate(entities)}
    n_nodes = len(entities)

    # Build node feature matrix (zero fallback for unknowns)
    x_rows = []
    n_missing = 0
    for entity in entities:
        vec = entity_embeddings.get(entity)
        if vec is None:
            n_missing += 1
            vec = np.zeros(embedding_dim, dtype=np.float32)
        x_rows.append(vec.astype(np.float32))

    if n_missing:
        logger.warning(
            "%d / %d subgraph entities not in embedding index; using zero vectors.",
            n_missing,
            n_nodes,
        )

    x = torch.tensor(np.stack(x_rows), dtype=torch.float)

    # Build edge_index
    src = [node_index[s] for s, _, _ in triples]
    dst = [node_index[o] for _, _, o in triples]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
    # Attach metadata for pipeline use
    data.node_index = node_index  # type: ignore[assignment]
    data.entities = entities  # type: ignore[assignment]

    logger.debug("subgraph_to_pyg: %d nodes, %d edges.", n_nodes, edge_index.shape[1])
    return data


def subgraph_to_networkx(
    triples: list[tuple[str, str, str]],
) -> nx.DiGraph:
    """Convert a list of triples to a NetworkX DiGraph.

    Used by GraphRAGPipeline for Louvain community detection
    (python-louvain works on undirected NetworkX graphs).

    Args:
        triples: List of (subject, relation, object) tuples.

    Returns:
        nx.DiGraph with entity strings as node labels and relation as
        edge attribute "relation".
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required. Run: pip install networkx") from exc

    g = nx.DiGraph()
    for subject, relation, obj in triples:
        g.add_edge(subject, obj, relation=relation)
    return g


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _count_unique_entities(triples: list[tuple[str, str, str]]) -> int:
    """Count unique entities in a triple list."""
    return len({s for s, _, _ in triples} | {o for _, _, o in triples})


def _trim_by_degree(
    triples: list[tuple[str, str, str]], max_nodes: int
) -> list[tuple[str, str, str]]:
    """Trim a subgraph to the max_nodes highest-degree entities.

    Counts combined in+out degree and retains top-max_nodes entities,
    then filters triples to those where both endpoints are retained.

    Args:
        triples: Original triple list.
        max_nodes: Maximum number of entities to retain.

    Returns:
        Filtered triple list with at most max_nodes unique entities.
    """
    degree: Counter = Counter()
    for s, _, o in triples:
        degree[s] += 1
        degree[o] += 1

    top_entities = {e for e, _ in degree.most_common(max_nodes)}
    return [(s, r, o) for s, r, o in triples if s in top_entities and o in top_entities]
