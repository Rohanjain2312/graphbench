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
                   │  50k triples    │
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
               │  (4-bit quantized)   │
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

TBD — updated after Phase 4 implementation.
