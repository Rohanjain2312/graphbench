"""GraphBench: Benchmark framework for graph-based RAG pipelines on multi-hop QA.

Compares GraphRAG (Louvain community detection) vs GNN-RAG (3-layer GAT)
on 500 HotpotQA distractor questions backed by a 50k-triple Neo4j knowledge graph.
"""

from graphbench.benchmark.evaluator import Evaluator
from graphbench.pipelines.base import Pipeline, PipelineResult
from graphbench.pipelines.gnnrag_pipeline import GNNRAGPipeline
from graphbench.pipelines.graphrag_pipeline import GraphRAGPipeline
from graphbench.version import __version__

__all__ = [
    "__version__",
    "Pipeline",
    "PipelineResult",
    "GraphRAGPipeline",
    "GNNRAGPipeline",
    "Evaluator",
]
