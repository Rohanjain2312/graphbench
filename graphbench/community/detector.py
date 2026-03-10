"""Louvain community detection on knowledge graph subgraphs.

Runs the Louvain algorithm (python-louvain / community library) on a
NetworkX graph extracted from Neo4j, with resolution=0.8.

Returns a mapping {node_id: community_id} and groups triples by community.

The top-3 communities most relevant to the input question are selected
(measured by entity overlap after embedding similarity). This forms the
context for GraphRAG's LLM prompt.

Implementation: Phase 4 (pipelines) — used by GraphRAGPipeline.
"""
