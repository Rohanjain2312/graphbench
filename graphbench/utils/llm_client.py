"""LLM inference client for GraphBench.

Abstracts over two backends depending on environment:
- **Colab / GPU**: HuggingFace Transformers with Mistral-7B-Instruct-v0.2
  loaded in 4-bit quantization via bitsandbytes.
- **Local Mac (dev)**: Phi-3-mini via Ollama HTTP API (no GPU required).

Both backends accept a formatted prompt string and return a decoded answer
string. The caller (Pipeline.answer) is responsible for formatting the prompt
using PROMPT_TEMPLATE from pipelines/base.py.

Implementation: Phase 4 (pipelines).
"""
