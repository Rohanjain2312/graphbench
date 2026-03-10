"""3-layer Graph Attention Network (GAT) model for edge relevance scoring.

Architecture:
- 3 GATConv layers (torch_geometric), 4 attention heads each
- Input: 384-dim entity embeddings (from all-MiniLM-L6-v2)
- Hidden: 256-dim (after first layer), 128-dim (after second layer)
- Output: edge relevance score in [0, 1] via sigmoid

Used by GNNRAGPipeline to rank subgraph edges by their relevance
to the query question. Training target: link prediction on the KG.

Must achieve test AUC-ROC > 0.75 before Phase 4 pipeline integration.

Implementation: Phase 3 (GNN).
"""
