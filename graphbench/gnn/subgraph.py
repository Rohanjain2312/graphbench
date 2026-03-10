"""Subgraph extraction utilities for GNN inference and pipeline retrieval.

Given a set of seed entity IDs, extracts a 2-hop ego-subgraph from Neo4j
and converts it to a PyG Data object ready for GAT forward pass.

Used by both pipelines (GraphRAG uses this for community detection input;
GNN-RAG uses this as the graph on which GAT scores edges).

Subgraph extraction parameters:
- hops = 2 (from seed entities)
- Max nodes per subgraph: 500 (trimmed by degree if exceeded)

Implementation: Phase 3 (GNN) and Phase 4 (pipelines).
"""
