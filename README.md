# GraphBench

[![CI](https://github.com/Rohanjain2312/graphbench/actions/workflows/tests.yml/badge.svg)](https://github.com/Rohanjain2312/graphbench/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/graphbench-kg)](https://pypi.org/project/graphbench-kg/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohanjain2312/graphbench/blob/main/notebooks/graphbench_babelscape.ipynb)

**GraphBench** is an open-source Python library that benchmarks two graph-based RAG
pipelines head-to-head on multi-hop question answering.

| Pipeline | Approach |
|----------|----------|
| **GraphRAG** | Louvain community detection + Mistral-7B |
| **GNN-RAG** | 3-layer GAT (PyTorch Geometric) + Mistral-7B |

Dataset: 500 HotpotQA distractor questions (250 bridge, 250 comparison)
Knowledge Graph: ~60k REBEL triples (Babelscape/rebel-dataset) in Neo4j AuraDB

---

## Notebook

The full end-to-end pipeline — data ingestion, GNN training, benchmarking, and results
analysis — runs in a single Google Colab notebook with all output cells pre-populated.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohanjain2312/graphbench/blob/main/notebooks/graphbench_babelscape.ipynb)

> Recruiters: click the badge above to view the fully executed notebook with results —
> no account or GPU required to read the outputs.

---

## Benchmark Results

500 HotpotQA distractor questions · seed=42 · Mistral-7B-Instruct-v0.2 (fp16) · 60k REBEL triples

| Metric | GraphRAG | GNN-RAG |
|--------|:--------:|:-------:|
| Exact Match (EM) | 3.2% | **5.0%** |
| Token F1 | 10.5% | **12.8%** |
| Latency p50 (ms) | 5,555 | **5,344** |
| Latency p95 (ms) | 6,388 | **6,199** |

GNN-RAG outperforms GraphRAG on all four metrics. The gap is largest on bridge questions
(7.2% vs 3.6% EM), where multi-hop graph traversal benefits most from learned edge scoring.

Full results and per-type breakdown: [docs/benchmark_results.md](docs/benchmark_results.md)

---

## Installation

```bash
pip install graphbench-kg
```

Or from source:

```bash
git clone https://github.com/Rohanjain2312/graphbench.git
cd graphbench
poetry install
```

---

## Quick Start

```python
from graphbench.utils.neo4j_client import Neo4jClient
from graphbench.utils.faiss_client import FAISSClient
from graphbench.utils.llm_client import LLMClient
from graphbench.pipelines.graphrag_pipeline import GraphRAGPipeline
from graphbench.pipelines.gnnrag_pipeline import GNNRAGPipeline
from graphbench.gnn.model import GATModel
from graphbench.utils.checkpoint import load_checkpoint
from pathlib import Path

# Init shared clients
neo4j = Neo4jClient()          # reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD from .env
faiss = FAISSClient.load()     # reads FAISS_INDEX_PATH from .env
llm   = LLMClient(backend="hf")

# GraphRAG pipeline
graphrag = GraphRAGPipeline(neo4j_client=neo4j, faiss_client=faiss, llm_client=llm)
print(graphrag.answer("Where was Marie Curie born?").predicted_answer)

# GNN-RAG pipeline (requires trained checkpoint + entity embeddings)
ckpt = load_checkpoint(Path("checkpoints/gat_best.pt"), map_location="cuda")
model = GATModel()
model.load_state_dict(ckpt["model_state_dict"])
gnnrag = GNNRAGPipeline(neo4j_client=neo4j, faiss_client=faiss, llm_client=llm,
                         gat_model=model, entity_embeddings=embedding_dict)
print(gnnrag.answer("Where was Marie Curie born?").predicted_answer)
```

---

## Architecture

![Architecture Diagram](assets/architecture_diagram.png)

> See [docs/architecture.md](docs/architecture.md) for full details.

---

## Project Structure

```
graphbench/
├── graphbench/          # Core library
│   ├── ingestion/       # REBEL loading, triple extraction, FAISS/Neo4j writers
│   ├── pipelines/       # GraphRAG and GNN-RAG pipelines
│   ├── gnn/             # GAT model, dataset, trainer
│   ├── community/       # Louvain community detection and summarization
│   ├── benchmark/       # HotpotQA loader, evaluator, metrics
│   └── utils/           # Neo4j client, FAISS client, LLM client, config
├── notebooks/           # End-to-end Colab notebooks (with pre-run outputs)
├── tests/               # Pytest test suite
└── docs/                # Documentation
```

---

## Requirements

- Python 3.10+
- Neo4j AuraDB Free (or local Neo4j)
- GPU recommended for GNN training and LLM inference (Google Colab Pro)
- See `.env.example` for required environment variables

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

[MIT](LICENSE) © 2025 Rohan Jain
