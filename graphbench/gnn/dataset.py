"""PyTorch Geometric dataset builder for GNN link-prediction training.

Converts the Neo4j knowledge graph into a PyG HeteroData / Data object:
- Nodes: entities with 384-dim embeddings as node features
- Edges: positive triples from the KG + randomly sampled negative edges
- 80/10/10 train/val/test split (stratified, no data leakage)

Negative sampling: for each positive edge, sample one negative by
corrupting either the head or tail entity uniformly at random.

Implementation: Phase 3 (GNN).
"""
