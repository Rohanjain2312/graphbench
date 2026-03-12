# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# GraphBench — CLAUDE.md

## What This Project Is
GraphBench is an open-source Python library (graphbench-kg) that benchmarks
two graph-based RAG pipelines head-to-head on multi-hop question answering:
- Pipeline A: GraphRAG (Louvain community detection + Mistral 7B)
- Pipeline B: GNN-RAG (3-layer GAT via PyTorch Geometric + Mistral 7B)
Benchmark dataset: 500 HotpotQA distractor questions (250 bridge, 250 comparison)
Knowledge graph: 50k REBEL triples stored in Neo4j AuraDB Free

## Repo & Paths
- Local: /Users/rohanjain/Desktop/UMD - MSML/Sem 4/graphbench
- GitHub: https://github.com/Rohanjain2312/graphbench.git
- Repo and local folder are already linked. NEVER run git init or git remote add.

## Stack
- Python 3.10, Poetry (package manager)
- PyTorch Geometric — GAT model (3 layers, 4 heads)
- Neo4j AuraDB Free — graph storage (text only, no embeddings)
- FAISS — vector index (384-dim, IndexFlatIP, separate from Neo4j)
- sentence-transformers/all-MiniLM-L6-v2 — embeddings
- mistralai/Mistral-7B-Instruct-v0.2 — LLM (Colab only, 4-bit quantized)
- Phi-3-mini via Ollama — LLM for local Mac dev only
- Weights & Biases — experiment tracking
- Gradio — HuggingFace Spaces demo (3 tabs)
- $0 API budget — 100% open source, no paid APIs

## Non-Negotiable Conventions
- Black formatting, line length 88
- isort for imports, ruff for linting
- Google-style docstrings on ALL public classes and functions
- Type hints on ALL function signatures
- NEVER hardcode secrets — all credentials via .env loaded through
  graphbench/utils/config.py (pydantic BaseSettings)
- NEVER commit .env, *.bin, *.pt, *.parquet, checkpoints/ (covered by .gitignore)
- NEVER create files outside the local project path above
- All heavy compute (GNN training, benchmark runs) → Colab Pro, not local Mac
- All library code uses Python logging, NOT print statements
- Notebook cells use print + tqdm (user-facing), NOT logging

## Project Structure (key files)
graphbench/utils/config.py     → single source of truth for all settings
graphbench/pipelines/base.py   → shared LLM prompt template lives here
graphbench/benchmark/metrics.py → normalize_answer() is the single source
                                   of truth — import it, never reimplement
notebooks/graphbench_babelscape.ipynb → PRIMARY notebook, all phases, Drive checkpointed
                                   notebooks/graphbench_main.ipynb → earlier draft, superseded

## Common Commands
poetry install                 → install all dependencies
poetry run pytest tests/       → run test suite
poetry run pytest --cov=graphbench tests/  → with coverage
poetry run black .             → format code
poetry run isort .             → sort imports
poetry run ruff check .        → lint
poetry build                   → build dist/ for PyPI

## External Services & Env Vars
NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD  → Neo4j AuraDB Free
HF_TOKEN                                   → HuggingFace Hub
WANDB_API_KEY, WANDB_PROJECT               → Weights & Biases
EMBEDDING_MODEL, LLM_MODEL                 → model name overrides
FAISS_INDEX_PATH, CHECKPOINT_DIR           → path overrides

## Phase Status (update this as you complete phases)
- Phase 1 — Foundation:        [x] COMPLETE
- Phase 2 — Data Pipeline:     [x] COMPLETE
- Phase 3 — GNN:               [x] COMPLETE
- Phase 4 — Pipelines:         [x] COMPLETE
- Phase 5 — Benchmark:         [x] COMPLETE
- Phase 6 — Notebook:          [x] COMPLETE
- Phase 7 — Demo:              [x] COMPLETE
- Phase 8 — Polish:            [x] COMPLETE

## Important Decisions Already Made (do not re-debate these)
- FAISS for vector search, Neo4j for graph traversal only (not Neo4j vector index)
- Top-50 REBEL relation types only (covers ~85% of triples)
- HotpotQA distractor setting, 500 questions, balanced bridge/comparison
- Both pipelines use IDENTICAL LLM prompt template from base.py
- Latency measured with time.perf_counter() around pipeline.answer() ONLY
- GNN test AUC-ROC must exceed 0.75 before proceeding to benchmark
- Community detection resolution=0.8, Louvain algorithm
- Subgraph extraction: 2 hops from seed entities
