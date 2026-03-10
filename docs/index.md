# GraphBench Documentation

GraphBench is an open-source Python library for benchmarking graph-based RAG pipelines
on multi-hop question answering.

## Contents

- [Quickstart](quickstart.md)
- [Architecture](architecture.md)
- [Benchmark Results](benchmark_results.md)

## Overview

GraphBench compares two pipelines head-to-head:

| Pipeline | Retrieval Strategy |
|----------|--------------------|
| GraphRAG | Louvain community detection (resolution=0.8) |
| GNN-RAG  | 3-layer GAT edge scoring (PyTorch Geometric) |

Both pipelines use the same LLM (Mistral-7B-Instruct-v0.2) and the same
prompt template to ensure a fair comparison.
