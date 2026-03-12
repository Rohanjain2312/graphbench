# Quickstart

## Installation

```bash
pip install graphbench-kg
```

## Environment Setup

```bash
cp .env.example .env
# Edit .env with your Neo4j and HuggingFace credentials
```

## Basic Usage

```python
from graphbench.utils.neo4j_client import Neo4jClient
from graphbench.utils.faiss_client import FAISSClient
from graphbench.utils.llm_client import LLMClient
from graphbench.pipelines.graphrag_pipeline import GraphRAGPipeline
from graphbench.pipelines.gnnrag_pipeline import GNNRAGPipeline
from graphbench.gnn.model import GATModel
from graphbench.utils.checkpoint import load_checkpoint
from pathlib import Path

# Init shared clients (reads credentials from .env)
neo4j = Neo4jClient()
faiss = FAISSClient.load()
llm   = LLMClient(backend="hf")

# GraphRAG pipeline
graphrag = GraphRAGPipeline(neo4j_client=neo4j, faiss_client=faiss, llm_client=llm)
result = graphrag.answer("Where was Marie Curie born?")
print(result.predicted_answer)

# GNN-RAG pipeline (requires a trained checkpoint and entity embeddings)
ckpt = load_checkpoint(Path("checkpoints/gat_best.pt"), map_location="cuda")
model = GATModel()
model.load_state_dict(ckpt["model_state_dict"])
gnnrag = GNNRAGPipeline(
    neo4j_client=neo4j,
    faiss_client=faiss,
    llm_client=llm,
    gat_model=model,
    entity_embeddings=embedding_dict,  # dict[str, np.ndarray]
)
result = gnnrag.answer("Where was Marie Curie born?")
print(result.predicted_answer)
```

## Running the Benchmark

```python
from graphbench.benchmark.evaluator import Evaluator

evaluator = Evaluator(pipelines=[graphrag, gnnrag], n_questions=500, seed=42)
results_df = evaluator.run()
print(results_df[["pipeline", "em", "f1", "latency_p50_ms"]])
```

Or from the Colab notebook which has all output cells pre-populated:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohanjain2312/graphbench/blob/main/notebooks/graphbench_babelscape.ipynb)

## Running the Demo

The full end-to-end pipeline runs in a single Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohanjain2312/graphbench/blob/main/notebooks/graphbench_babelscape.ipynb)

The notebook covers: data ingestion → GNN training → benchmark run → results analysis,
with all output cells pre-populated so you can read the results without running anything.
