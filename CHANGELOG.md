# Changelog

All notable changes to GraphBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
