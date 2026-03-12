# Changelog

All notable changes to GraphBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-12

### Fixed
- `GNNRAGPipeline.answer()`: move subgraph tensors to model device before GAT
  forward pass — fixes `RuntimeError: Expected all tensors on the same device`
  when model is on CUDA and subgraph PyG Data is on CPU
- `neo4j_client.py`: use f-string literal for hop count in Cypher path pattern
  (`[*1..{k}]`) — Neo4j rejects query parameters inside hop-count brackets
- `numpy` dependency constraint widened from `^1.26` to `>=1.26` — the old
  constraint blocked numpy 2.x, causing an ABI mismatch crash on Colab Python 3.12
  (`ValueError: numpy.dtype size changed`)
- `bitsandbytes` made optional (`pip install graphbench-kg[bnb]`) — 4-bit
  quantization is not required; fp16 loading works without it on modern GPUs

### Changed
- Knowledge graph source updated to `Babelscape/rebel-dataset` (60k pre-extracted
  REBEL triples) — replaces slow on-the-fly REBEL inference on HotpotQA passages
- Benchmark results documented: GNN-RAG 5.0% EM / 12.8% F1 vs GraphRAG 3.2% / 10.5%

## [0.1.0] - 2025-03-09

### Added
- Initial project scaffold and directory structure
- `graphbench/utils/config.py` — pydantic BaseSettings configuration
- `graphbench/pipelines/base.py` — `PipelineResult` dataclass and `Pipeline` ABC
- Module stubs for ingestion, gnn, community, benchmark, and utils
- Full test scaffold with pytest
- GitHub Actions CI (tests.yml) and PyPI publish workflow (publish.yml)
- Gradio demo skeleton
- Documentation skeleton (index, quickstart, architecture, benchmark_results)
