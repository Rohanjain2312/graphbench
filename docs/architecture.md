# Architecture

## System Overview

```
HotpotQA Question
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│  Embedder   │────▶│   FAISS Index    │ (top-10 seed entities)
│ all-MiniLM  │     │  384-dim IVF     │
└─────────────┘     └──────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Neo4j AuraDB   │ (2-hop subgraph)
                   │  60k triples    │
                   └────────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
   ┌──────────────────┐       ┌──────────────────────┐
   │   Pipeline A     │       │   Pipeline B         │
   │   GraphRAG       │       │   GNN-RAG            │
   │ Louvain (r=0.8)  │       │ 3-layer GAT (4 heads)│
   └────────┬─────────┘       └──────────┬───────────┘
            │                            │
            └─────────────┬──────────────┘
                          ▼
               ┌─────────────────────┐
               │  Mistral-7B-Instruct │
               │  (fp16, Colab A100)  │
               │  Shared PROMPT_TMPL  │
               └──────────┬──────────┘
                          ▼
                    Predicted Answer
                          │
                          ▼
              ┌───────────────────────┐
              │  Evaluator            │
              │  EM, F1, Latency p50  │
              │  W&B logging          │
              └───────────────────────┘
```

## Key Design Decisions

- **FAISS only for vector search** — Neo4j is graph traversal only, never vector index
- **Shared LLM prompt** — both pipelines use the identical PROMPT_TEMPLATE from base.py
- **Latency measurement** — time.perf_counter() around pipeline.answer() only
- **GNN gate** — GNN must achieve test AUC-ROC > 0.75 before benchmark runs

## Component Details

| Component | Class | Location | Role |
|-----------|-------|----------|------|
| **Neo4jClient** | `Neo4jClient` | `graphbench/utils/neo4j_client.py` | Executes Cypher queries against Neo4j AuraDB Free to extract 2-hop subgraphs around seed entities |
| **FAISSClient** | `FAISSClient` | `graphbench/utils/faiss_client.py` | Loads a 384-dim `IndexFlatIP` FAISS index; provides `search(vec, k)` for top-K entity lookup |
| **LLMClient** | `LLMClient` | `graphbench/utils/llm_client.py` | Unified client over Ollama (local Mac dev, Phi-3-mini) and HuggingFace Transformers (Colab GPU, Mistral-7B). Auto-detects the backend. |
| **CommunityDetector** | `CommunityDetector` | `graphbench/community/detector.py` | Runs Louvain community detection (resolution=0.8) on a NetworkX graph; selects top communities by seed-entity overlap |
| **GATModel** | `GATModel` | `graphbench/gnn/model.py` | 3-layer GAT encoder (384→256→64→32) with dot-product decoder for link-prediction edge scoring |
| **Evaluator** | `Evaluator` | `graphbench/benchmark/evaluator.py` | Runs both pipelines on HotpotQA questions; computes EM, token F1, and latency; logs to W&B |
