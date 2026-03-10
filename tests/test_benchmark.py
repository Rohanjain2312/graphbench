"""Tests for graphbench.benchmark (Phase 5).

Covers:
- load_hotpotqa() — mocked HuggingFace dataset
- Evaluator.run() — mocked pipelines, metric computation, latency tracking
- LLMJudge.judge() — verdict parsing, batch judging
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphbench.benchmark.evaluator import Evaluator
from graphbench.benchmark.llm_judge import LLMJudge
from graphbench.pipelines.base import PipelineResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_question(
    qid: str = "q1",
    question: str = "Where was Einstein born?",
    answer: str = "Ulm",
    qtype: str = "bridge",
) -> dict:
    return {"id": qid, "question": question, "answer": answer, "type": qtype}


def _make_mock_pipeline(name: str, predicted_answer: str = "Ulm") -> MagicMock:
    """Build a mock Pipeline whose answer() returns a fixed PipelineResult."""
    pipeline = MagicMock()
    pipeline.name = name
    pipeline.answer.return_value = PipelineResult(
        question="Where was Einstein born?",
        predicted_answer=predicted_answer,
        pipeline_name=name,
        context_triples=[("Einstein", "born_in", "Ulm")],
    )
    return pipeline


# ---------------------------------------------------------------------------
# load_hotpotqa
# ---------------------------------------------------------------------------


class TestLoadHotpotQA:
    """Tests for graphbench.benchmark.hotpotqa_loader.load_hotpotqa."""

    def _make_fake_dataset(self, n_bridge: int = 300, n_comparison: int = 300):
        """Build a fake HotpotQA dataset list for mocking."""
        rows = []
        for i in range(n_bridge):
            rows.append(
                {
                    "id": f"b{i}",
                    "question": f"Bridge Q {i}",
                    "answer": f"A{i}",
                    "type": "bridge",
                }
            )
        for i in range(n_comparison):
            rows.append(
                {
                    "id": f"c{i}",
                    "question": f"Comparison Q {i}",
                    "answer": f"B{i}",
                    "type": "comparison",
                }
            )
        return rows

    def test_returns_correct_count(self) -> None:
        """load_hotpotqa(n=100) should return exactly 100 questions."""
        from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

        fake_ds = self._make_fake_dataset()
        with patch(
            "datasets.load_dataset",
            return_value=fake_ds,
        ):
            questions = load_hotpotqa(n=100)
        assert len(questions) == 100

    def test_balanced_types(self) -> None:
        """load_hotpotqa() should return equal bridge and comparison counts."""
        from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

        fake_ds = self._make_fake_dataset()
        with patch(
            "datasets.load_dataset",
            return_value=fake_ds,
        ):
            questions = load_hotpotqa(n=100)

        n_bridge = sum(1 for q in questions if q["type"] == "bridge")
        n_comparison = sum(1 for q in questions if q["type"] == "comparison")
        assert n_bridge == 50
        assert n_comparison == 50

    def test_required_keys_present(self) -> None:
        """Each question dict should have id, question, answer, type."""
        from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

        fake_ds = self._make_fake_dataset()
        with patch(
            "datasets.load_dataset",
            return_value=fake_ds,
        ):
            questions = load_hotpotqa(n=10)

        for q in questions:
            assert "id" in q
            assert "question" in q
            assert "answer" in q
            assert "type" in q

    def test_deterministic_with_same_seed(self) -> None:
        """Two calls with same seed should return identical question sets."""
        from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

        fake_ds = self._make_fake_dataset()
        with patch(
            "datasets.load_dataset",
            return_value=fake_ds,
        ):
            q1 = load_hotpotqa(n=100, seed=42)
            q2 = load_hotpotqa(n=100, seed=42)

        assert [q["id"] for q in q1] == [q["id"] for q in q2]

    def test_different_seeds_give_different_samples(self) -> None:
        """Different seeds should (with high probability) give different sets."""
        from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

        fake_ds = self._make_fake_dataset(500, 500)
        with patch(
            "datasets.load_dataset",
            return_value=fake_ds,
        ):
            q1 = load_hotpotqa(n=100, seed=42)
            q2 = load_hotpotqa(n=100, seed=99)

        ids1 = {q["id"] for q in q1}
        ids2 = {q["id"] for q in q2}
        assert ids1 != ids2

    def test_raises_on_odd_n(self) -> None:
        """load_hotpotqa() should raise ValueError for odd n."""
        from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

        with pytest.raises(ValueError, match="even"):
            load_hotpotqa(n=101)

    def test_raises_on_zero_n(self) -> None:
        """load_hotpotqa() should raise ValueError for n=0."""
        from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

        with pytest.raises(ValueError):
            load_hotpotqa(n=0)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    """Tests for graphbench.benchmark.evaluator.Evaluator."""

    def _run_evaluator(
        self,
        pred_a: str = "Ulm",
        pred_b: str = "Warsaw",
        gold: str = "Ulm",
        n: int = 4,
    ) -> dict:
        """Run Evaluator with mocked pipelines and dataset."""
        pipeline_a = _make_mock_pipeline("GraphRAG", predicted_answer=pred_a)
        pipeline_b = _make_mock_pipeline("GNN-RAG", predicted_answer=pred_b)

        # Override answer() return to use the supplied question from the loader
        def side_effect_a(question):
            return PipelineResult(
                question=question,
                predicted_answer=pred_a,
                pipeline_name="GraphRAG",
                gold_answer=gold,
            )

        def side_effect_b(question):
            return PipelineResult(
                question=question,
                predicted_answer=pred_b,
                pipeline_name="GNN-RAG",
                gold_answer=gold,
            )

        pipeline_a.answer.side_effect = side_effect_a
        pipeline_b.answer.side_effect = side_effect_b

        fake_questions = [
            _make_question(
                qid=f"q{i}", answer=gold, qtype="bridge" if i < n // 2 else "comparison"
            )
            for i in range(n)
        ]

        evaluator = Evaluator(
            pipeline_a=pipeline_a,
            pipeline_b=pipeline_b,
            n_questions=n,
            use_wandb=False,
        )
        with (
            patch(
                "graphbench.benchmark.evaluator.load_hotpotqa",
                return_value=fake_questions,
            ),
            patch.object(evaluator, "_save_results"),
        ):  # skip disk writes
            return evaluator.run()

    def test_run_returns_both_pipeline_names(self) -> None:
        """run() should return a dict keyed by both pipeline names."""
        summary = self._run_evaluator()
        assert "GraphRAG" in summary
        assert "GNN-RAG" in summary

    def test_run_returns_required_metric_keys(self) -> None:
        """Each pipeline entry should have em, f1, latency_p50, latency_p95."""
        summary = self._run_evaluator()
        for key in ("em", "f1", "latency_p50", "latency_p95", "n_questions"):
            assert key in summary["GraphRAG"], f"Missing key: {key}"
            assert key in summary["GNN-RAG"], f"Missing key: {key}"

    def test_perfect_predictions_give_em_1(self) -> None:
        """When predicted == gold, EM should be 1.0."""
        summary = self._run_evaluator(pred_a="Ulm", pred_b="Ulm", gold="Ulm")
        assert abs(summary["GraphRAG"]["em"] - 1.0) < 1e-6
        assert abs(summary["GNN-RAG"]["em"] - 1.0) < 1e-6

    def test_wrong_predictions_give_em_0(self) -> None:
        """When predicted != gold, EM should be 0.0."""
        summary = self._run_evaluator(pred_a="Berlin", pred_b="Berlin", gold="Ulm")
        assert abs(summary["GraphRAG"]["em"] - 0.0) < 1e-6
        assert abs(summary["GNN-RAG"]["em"] - 0.0) < 1e-6

    def test_latency_is_positive(self) -> None:
        """Latency metrics should be >= 0."""
        summary = self._run_evaluator()
        for name in ("GraphRAG", "GNN-RAG"):
            assert summary[name]["latency_p50"] >= 0.0
            assert summary[name]["latency_p95"] >= 0.0

    def test_n_questions_matches(self) -> None:
        """n_questions in summary should match the requested count."""
        summary = self._run_evaluator(n=4)
        assert summary["GraphRAG"]["n_questions"] == 4
        assert summary["GNN-RAG"]["n_questions"] == 4

    def test_pipeline_error_does_not_crash_evaluator(self) -> None:
        """If a pipeline raises, run() should log the error and continue."""
        pipeline_a = _make_mock_pipeline("GraphRAG")
        pipeline_b = _make_mock_pipeline("GNN-RAG")
        pipeline_a.answer.side_effect = RuntimeError("Simulated failure")

        fake_questions = [_make_question(qid=f"q{i}") for i in range(4)]

        evaluator = Evaluator(
            pipeline_a=pipeline_a,
            pipeline_b=pipeline_b,
            n_questions=4,
            use_wandb=False,
        )
        with (
            patch(
                "graphbench.benchmark.evaluator.load_hotpotqa",
                return_value=fake_questions,
            ),
            patch.object(evaluator, "_save_results"),
        ):
            summary = evaluator.run()  # should not raise

        assert "GraphRAG" in summary


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """Tests for graphbench.benchmark.llm_judge.LLMJudge."""

    def _make_judge(self, llm_response: str) -> LLMJudge:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_response
        return LLMJudge(llm_client=mock_llm)

    def test_judge_returns_bool_and_str(self) -> None:
        """judge() should return (bool, str)."""
        judge = self._make_judge("VERDICT: correct\nThe answer matches.")
        result = judge.judge("Q?", "Ulm", "Ulm")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_judge_correct_verdict(self) -> None:
        """judge() should return True for 'VERDICT: correct'."""
        judge = self._make_judge("VERDICT: correct\nMatches exactly.")
        is_correct, _ = judge.judge("Q?", "Ulm", "Ulm")
        assert is_correct is True

    def test_judge_incorrect_verdict(self) -> None:
        """judge() should return False for 'VERDICT: incorrect'."""
        judge = self._make_judge("VERDICT: incorrect\nDoes not match.")
        is_correct, _ = judge.judge("Q?", "Berlin", "Ulm")
        assert is_correct is False

    def test_judge_case_insensitive_verdict(self) -> None:
        """judge() should be case-insensitive for VERDICT keyword."""
        judge = self._make_judge("verdict: CORRECT\nOK")
        is_correct, _ = judge.judge("Q?", "Ulm", "Ulm")
        assert is_correct is True

    def test_judge_unparseable_defaults_false(self) -> None:
        """judge() should default to False if response cannot be parsed."""
        judge = self._make_judge("I am confused and cannot decide.")
        is_correct, _ = judge.judge("Q?", "Ulm", "Ulm")
        assert is_correct is False

    def test_judge_llm_error_returns_false(self) -> None:
        """judge() should return (False, error_str) if LLM raises."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM offline")
        judge = LLMJudge(llm_client=mock_llm)
        is_correct, reason = judge.judge("Q?", "x", "y")
        assert is_correct is False
        assert "LLM" in reason or "error" in reason.lower()

    def test_judge_batch_length(self) -> None:
        """judge_batch() should return one result per question."""
        judge = self._make_judge("VERDICT: correct\nOK")
        results = judge.judge_batch(
            questions=["Q1?", "Q2?", "Q3?"],
            predicted_answers=["A", "B", "C"],
            gold_answers=["A", "B", "C"],
        )
        assert len(results) == 3

    def test_judge_batch_raises_on_length_mismatch(self) -> None:
        """judge_batch() should raise ValueError if list lengths differ."""
        judge = self._make_judge("VERDICT: correct")
        with pytest.raises(ValueError):
            judge.judge_batch(["Q1", "Q2"], ["A"], ["A", "B"])
