"""Main benchmark evaluator for comparing GraphRAG vs GNN-RAG.

Orchestrates the full benchmark run:
1. Load 500 HotpotQA questions via hotpotqa_loader.
2. For each question, call pipeline.answer() for both pipelines.
3. Measure latency with time.perf_counter() around each answer() call.
4. Compute Exact Match and F1 via metrics.py (normalize_answer is canonical).
5. Log per-question results to W&B table.
6. Produce summary statistics: mean EM, mean F1, p50/p95 latency by pipeline.

Results are saved to experiments/results/ as JSON and CSV.

Implementation: Phase 5 (benchmark).
"""

from graphbench.pipelines.base import Pipeline


class Evaluator:
    """Benchmark evaluator comparing two pipelines on HotpotQA.

    Args:
        pipeline_a: First pipeline (GraphRAG).
        pipeline_b: Second pipeline (GNN-RAG).
        n_questions: Number of questions to evaluate (default: 500).
        seed: Random seed for question sampling (default: 42).
    """

    def __init__(
        self,
        pipeline_a: Pipeline,
        pipeline_b: Pipeline,
        n_questions: int = 500,
        seed: int = 42,
    ) -> None:
        """Initialise the evaluator with two pipelines and benchmark settings."""
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b
        self.n_questions = n_questions
        self.seed = seed

    def run(self) -> dict:
        """Run the full benchmark and return results dict.

        Returns:
            Dictionary with per-pipeline metrics: em, f1, latency_p50, latency_p95.
        """
        # TODO: Phase 5 implementation
        raise NotImplementedError("Evaluator.run() — implement in Phase 5")
