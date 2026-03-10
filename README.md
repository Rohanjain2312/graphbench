# GraphBench

[![CI](https://github.com/Rohanjain2312/graphbench/actions/workflows/tests.yml/badge.svg)](https://github.com/Rohanjain2312/graphbench/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/graphbench-kg)](https://pypi.org/project/graphbench-kg/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Demo](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-orange)](https://huggingface.co/spaces/rohanjain2312/graphbench)

**GraphBench** is an open-source Python library that benchmarks two graph-based RAG
pipelines head-to-head on multi-hop question answering.

| Pipeline | Approach |
|----------|----------|
| **GraphRAG** | Louvain community detection + Mistral-7B |
| **GNN-RAG** | 3-layer GAT (PyTorch Geometric) + Mistral-7B |

Dataset: 500 HotpotQA distractor questions (250 bridge, 250 comparison)
Knowledge Graph: 50k REBEL triples in Neo4j AuraDB

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
# TBD — Phase 4
```

---

## Architecture

![Architecture Diagram](assets/architecture_diagram.png)

> See [docs/architecture.md](docs/architecture.md) for full details.

---

## Benchmark Results

> TBD — Phase 5

| Metric | GraphRAG | GNN-RAG |
|--------|----------|---------|
| Exact Match | TBD | TBD |
| F1 Score | TBD | TBD |
| Latency (p50) | TBD | TBD |

Full results: [docs/benchmark_results.md](docs/benchmark_results.md)

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
├── notebooks/           # End-to-end Colab notebook
├── demo/                # Gradio HuggingFace Spaces demo
├── tests/               # Pytest test suite
└── docs/                # Documentation
```

---

## Requirements

- Python 3.10+
- Neo4j AuraDB Free (or local Neo4j)
- GPU recommended for GNN training (Colab Pro)
- See `.env.example` for required environment variables

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

[MIT](LICENSE) © 2025 Rohan Jain
