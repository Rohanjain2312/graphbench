# Benchmark Results

## Setup

- **Dataset**: HotpotQA distractor setting, 500 questions (250 bridge + 250 comparison), seed=42
- **Knowledge Graph**: ~60k REBEL triples (Babelscape/rebel-dataset, TOP_50_RELATIONS filter) in Neo4j AuraDB
- **LLM**: mistralai/Mistral-7B-Instruct-v0.2, fp16, device_map=auto
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim, FAISS IndexFlatIP)
- **GNN**: 3-layer GAT (4 heads, hidden=256), test AUC-ROC=0.7697, trained 172 epochs

## Metrics

| Metric | Description |
|--------|-------------|
| Exact Match (EM) | 1.0 if normalised prediction == normalised gold answer |
| Token F1 | Token-level precision/recall harmonic mean |
| Latency p50 | Median wall-clock time per question (ms), measured around `pipeline.answer()` only |
| Latency p95 | 95th-percentile latency (ms) |

## Overall Results

| Metric | GraphRAG | GNN-RAG | Delta |
|--------|:--------:|:-------:|:-----:|
| Exact Match | 3.2% | **5.0%** | +1.8pp |
| Token F1 | 10.5% | **12.8%** | +2.3pp |
| Latency p50 (ms) | 5,555 | **5,344** | −211ms |
| Latency p95 (ms) | 6,388 | **6,199** | −189ms |
| N Questions | 500 | 500 | — |

## By Question Type

| Type | GraphRAG EM | GNN-RAG EM | Delta |
|------|:-----------:|:----------:|:-----:|
| Bridge | 3.6% | **7.2%** | +3.6pp |
| Comparison | 2.8% | **2.8%** | 0pp |

## Key Findings

- **GNN-RAG wins on all metrics**: +56% relative EM improvement (3.2% → 5.0%), +22% F1.
- **Bridge questions drive the gap**: GNN-RAG doubles EM on bridge questions (3.6% → 7.2%).
  Multi-hop bridge reasoning benefits from learned edge scoring — the GAT selects more
  relevant paths than Louvain community detection.
- **Comparison questions are a draw**: Both pipelines score 2.8% EM on comparison questions.
  Entity-level attribute comparison is harder to capture from graph triples alone.
- **GNN-RAG is also faster**: −211ms p50 latency despite the extra GAT forward pass.
  Community detection (Louvain) adds more overhead than GAT edge scoring on small subgraphs.
- **EM is low overall**: HotpotQA requires precise multi-hop reasoning; EM penalises any
  wording mismatch. Token F1 (10–13%) better reflects partial answer quality.

## Reproducibility

```bash
# Reproduce benchmark from scratch
poetry run python -m graphbench.ingestion.run_pipeline \
    --preextracted data/triples_babelscape.parquet

# Or use the Colab notebook (requires GPU for GNN training + LLM inference)
# notebooks/graphbench_babelscape.ipynb
```

W&B runs: https://wandb.ai/rohanjain2312-university-of-maryland/graphbench
