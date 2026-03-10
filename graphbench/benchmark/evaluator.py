"""Main benchmark evaluator for comparing GraphRAG vs GNN-RAG.

Orchestrates the full benchmark run:
1. Load N HotpotQA questions via :func:`~graphbench.benchmark.hotpotqa_loader.load_hotpotqa`.
2. For each question, call ``pipeline.answer()`` for both pipelines.
3. Measure latency with ``time.perf_counter()`` around each ``answer()`` call.
4. Compute Exact Match and F1 via :mod:`graphbench.benchmark.metrics`.
5. Log per-question results to a W&B table (if WANDB_API_KEY is set).
6. Produce summary statistics: mean EM, mean F1, p50/p95 latency per pipeline.

Results are saved to ``experiments/results/`` as JSON and CSV.

Usage::

    evaluator = Evaluator(pipeline_a=graphrag, pipeline_b=gnnrag)
    summary = evaluator.run()
    print(summary)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from graphbench.benchmark.hotpotqa_loader import load_hotpotqa
from graphbench.benchmark.metrics import exact_match, token_f1
from graphbench.pipelines.base import Pipeline, PipelineResult

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS_DIR = Path("experiments/results")


class Evaluator:
    """Benchmark evaluator comparing two pipelines on HotpotQA.

    Args:
        pipeline_a: First pipeline (GraphRAG).
        pipeline_b: Second pipeline (GNN-RAG).
        n_questions: Number of questions to evaluate (must be even, default: 500).
        seed: Random seed for question sampling (default: 42).
        results_dir: Directory for saving JSON/CSV results.
            Defaults to ``experiments/results/``.
        use_wandb: Whether to log to W&B. Defaults to True (skipped if no API key).
    """

    def __init__(
        self,
        pipeline_a: Pipeline,
        pipeline_b: Pipeline,
        n_questions: int = 500,
        seed: int = 42,
        results_dir: Path | None = None,
        use_wandb: bool = True,
    ) -> None:
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b
        self.n_questions = n_questions
        self.seed = seed
        self.results_dir = results_dir or _DEFAULT_RESULTS_DIR
        self.use_wandb = use_wandb

    def run(self) -> dict[str, Any]:
        """Run the full benchmark and return a summary results dict.

        For each question, both pipelines are called sequentially. Latency is
        measured with ``time.perf_counter()`` around ``pipeline.answer()`` only.

        Returns:
            Dict keyed by pipeline name, each value a dict with:
            ``em``, ``f1``, ``latency_p50``, ``latency_p95``,
            ``n_questions``, ``question_type_breakdown``.

        Raises:
            RuntimeError: If any pipeline raises unexpectedly during evaluation
                (per-question errors are caught and logged, not re-raised).
        """
        questions = load_hotpotqa(n=self.n_questions, seed=self.seed)
        logger.info(
            "Evaluator: running %d questions on '%s' vs '%s'.",
            len(questions),
            self.pipeline_a.name,
            self.pipeline_b.name,
        )

        results_a: list[PipelineResult] = []
        results_b: list[PipelineResult] = []

        wandb_run = self._init_wandb()

        try:
            from tqdm import tqdm  # noqa: PLC0415

            iterator = tqdm(questions, desc="Benchmark", unit="q")
        except ImportError:
            iterator = questions  # type: ignore[assignment]

        for q in iterator:
            result_a = self._run_one(self.pipeline_a, q)
            result_b = self._run_one(self.pipeline_b, q)
            results_a.append(result_a)
            results_b.append(result_b)

            if wandb_run is not None:
                self._log_row_wandb(wandb_run, q, result_a, result_b)

        summary = {
            self.pipeline_a.name: self._summarize(results_a),
            self.pipeline_b.name: self._summarize(results_b),
        }

        self._save_results(questions, results_a, results_b, summary)

        if wandb_run is not None:
            self._log_summary_wandb(wandb_run, summary)
            wandb_run.finish()

        logger.info("Benchmark complete. Summary:\n%s", json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------
    # Per-question evaluation
    # ------------------------------------------------------------------

    def _run_one(self, pipeline: Pipeline, q: dict) -> PipelineResult:
        """Call pipeline.answer() and attach latency + gold answer.

        Args:
            pipeline: The pipeline to evaluate.
            q: Question dict with keys ``question``, ``answer``, ``id``, ``type``.

        Returns:
            PipelineResult with ``latency_ms`` and ``gold_answer`` set.
        """
        t0 = time.perf_counter()
        try:
            result = pipeline.answer(q["question"])
        except Exception as exc:
            logger.error(
                "Pipeline '%s' raised on q_id=%s: %s",
                pipeline.name,
                q.get("id", "?"),
                exc,
                exc_info=True,
            )
            result = PipelineResult(
                question=q["question"],
                predicted_answer="",
                pipeline_name=pipeline.name,
                metadata={"error": str(exc)},
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result.latency_ms = latency_ms
        result.gold_answer = q["answer"]
        result.pipeline_name = pipeline.name
        return result

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize(results: list[PipelineResult]) -> dict[str, Any]:
        """Compute aggregate metrics for a list of PipelineResults.

        Args:
            results: All results for one pipeline across the benchmark.

        Returns:
            Dict with ``em``, ``f1``, ``latency_p50``, ``latency_p95``,
            ``n_questions``.
        """
        ems = [exact_match(r.predicted_answer, r.gold_answer or "") for r in results]
        f1s = [token_f1(r.predicted_answer, r.gold_answer or "") for r in results]
        latencies = [r.latency_ms for r in results]

        return {
            "em": float(np.mean(ems)),
            "f1": float(np.mean(f1s)),
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "n_questions": len(results),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_results(
        self,
        questions: list[dict],
        results_a: list[PipelineResult],
        results_b: list[PipelineResult],
        summary: dict[str, Any],
    ) -> None:
        """Save per-question results and summary to disk as JSON and CSV.

        Args:
            questions: Original question dicts.
            results_a: Results from pipeline A.
            results_b: Results from pipeline B.
            summary: Aggregate summary dict.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Build per-question rows
        rows = []
        for q, ra, rb in zip(questions, results_a, results_b):
            rows.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "type": q["type"],
                    f"{self.pipeline_a.name}_predicted": ra.predicted_answer,
                    f"{self.pipeline_a.name}_em": exact_match(
                        ra.predicted_answer, q["answer"]
                    ),
                    f"{self.pipeline_a.name}_f1": token_f1(
                        ra.predicted_answer, q["answer"]
                    ),
                    f"{self.pipeline_a.name}_latency_ms": ra.latency_ms,
                    f"{self.pipeline_b.name}_predicted": rb.predicted_answer,
                    f"{self.pipeline_b.name}_em": exact_match(
                        rb.predicted_answer, q["answer"]
                    ),
                    f"{self.pipeline_b.name}_f1": token_f1(
                        rb.predicted_answer, q["answer"]
                    ),
                    f"{self.pipeline_b.name}_latency_ms": rb.latency_ms,
                }
            )

        # JSON — full results
        json_path = self.results_dir / f"{ts}_results.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        logger.info("Saved JSON results → %s", json_path)

        # CSV — per-question table
        try:
            import pandas as pd  # noqa: PLC0415

            csv_path = self.results_dir / f"{ts}_results.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            logger.info("Saved CSV results → %s", csv_path)
        except ImportError:
            logger.warning("pandas not installed — skipping CSV export.")

    # ------------------------------------------------------------------
    # W&B integration
    # ------------------------------------------------------------------

    def _init_wandb(self):
        """Initialise a W&B run if WANDB_API_KEY is set and use_wandb=True."""
        if not self.use_wandb:
            return None
        try:
            from graphbench.utils.config import settings  # noqa: PLC0415

            if not settings.wandb_api_key:
                return None

            import wandb  # noqa: PLC0415

            run = wandb.init(
                project=settings.wandb_project,
                job_type="benchmark",
                config={
                    "n_questions": self.n_questions,
                    "seed": self.seed,
                    "pipeline_a": self.pipeline_a.name,
                    "pipeline_b": self.pipeline_b.name,
                },
            )
            logger.info("W&B benchmark run initialised: %s", run.name)
            return run
        except Exception as exc:
            logger.warning("W&B init failed: %s. Continuing without W&B.", exc)
            return None

    @staticmethod
    def _log_row_wandb(run, q: dict, ra: PipelineResult, rb: PipelineResult) -> None:
        """Log a single question's results to W&B."""
        try:

            run.log(
                {
                    "question_id": q["id"],
                    "question_type": q["type"],
                    f"{ra.pipeline_name}/em": exact_match(
                        ra.predicted_answer, q["answer"]
                    ),
                    f"{ra.pipeline_name}/f1": token_f1(
                        ra.predicted_answer, q["answer"]
                    ),
                    f"{ra.pipeline_name}/latency_ms": ra.latency_ms,
                    f"{rb.pipeline_name}/em": exact_match(
                        rb.predicted_answer, q["answer"]
                    ),
                    f"{rb.pipeline_name}/f1": token_f1(
                        rb.predicted_answer, q["answer"]
                    ),
                    f"{rb.pipeline_name}/latency_ms": rb.latency_ms,
                }
            )
        except Exception as exc:
            logger.debug("W&B row log failed: %s", exc)

    @staticmethod
    def _log_summary_wandb(run, summary: dict[str, Any]) -> None:
        """Log final summary metrics to W&B."""
        try:
            flat = {}
            for pipeline_name, metrics in summary.items():
                for key, val in metrics.items():
                    flat[f"{pipeline_name}/{key}"] = val
            run.summary.update(flat)
        except Exception as exc:
            logger.debug("W&B summary log failed: %s", exc)
