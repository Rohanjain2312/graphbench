"""LLM-based answer quality judge for qualitative benchmark analysis.

Uses an LLM (via :class:`~graphbench.utils.llm_client.LLMClient`) to judge
whether a predicted answer is semantically correct given the gold answer,
even when token-level F1 fails (e.g., paraphrases, date formats, entity aliases).

Output per question: ``(is_correct: bool, reasoning: str)``.

Used for qualitative analysis only — EM and F1 remain the primary metrics.
LLMJudge results are logged to W&B alongside EM/F1 but do NOT influence the
final benchmark ranking.

Usage::

    from graphbench.benchmark.llm_judge import LLMJudge
    from graphbench.utils.llm_client import LLMClient

    judge = LLMJudge(LLMClient())
    correct, reason = judge.judge("Where was Einstein born?", "Ulm", "Ulm, Germany")
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are an answer evaluation assistant.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Is the predicted answer correct? A predicted answer is correct if it conveys
the same core information as the gold answer, even if worded differently.
Partial matches (e.g., "Germany" when gold is "Ulm, Germany") are incorrect.

Respond with exactly one line in this format:
VERDICT: correct
or
VERDICT: incorrect
Then on a new line, give a brief (one sentence) reason.
"""


class LLMJudge:
    """LLM-based semantic correctness judge for benchmark analysis.

    Args:
        llm_client: Initialised :class:`~graphbench.utils.llm_client.LLMClient`.
            The same client used by the pipelines can be reused here.
    """

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def judge(
        self,
        question: str,
        predicted: str,
        gold: str,
    ) -> tuple[bool, str]:
        """Judge whether a predicted answer is semantically correct.

        Args:
            question: The original question string.
            predicted: The pipeline's predicted answer.
            gold: The gold (ground-truth) answer.

        Returns:
            ``(is_correct, reasoning)`` where ``is_correct`` is True if the LLM
            judges the predicted answer as correct, and ``reasoning`` is a brief
            explanation string.
        """
        prompt = _JUDGE_PROMPT.format(
            question=question,
            gold=gold,
            predicted=predicted,
        )
        try:
            response = self._llm.generate(prompt)
        except Exception as exc:
            logger.warning("LLMJudge.generate() failed: %s. Defaulting to False.", exc)
            return False, f"LLM error: {exc}"

        return self._parse_response(response)

    def judge_batch(
        self,
        questions: list[str],
        predicted_answers: list[str],
        gold_answers: list[str],
    ) -> list[tuple[bool, str]]:
        """Judge a batch of predictions.

        Args:
            questions: List of question strings.
            predicted_answers: List of predicted answer strings.
            gold_answers: List of gold answer strings.

        Returns:
            List of ``(is_correct, reasoning)`` tuples, one per question.
        """
        if not (len(questions) == len(predicted_answers) == len(gold_answers)):
            raise ValueError(
                "questions, predicted_answers, and gold_answers must have the same length."
            )
        return [
            self.judge(q, pred, gold)
            for q, pred, gold in zip(questions, predicted_answers, gold_answers)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: str) -> tuple[bool, str]:
        """Parse the LLM judge response into (is_correct, reasoning).

        Looks for ``VERDICT: correct`` or ``VERDICT: incorrect`` on the first
        line. Any subsequent text is treated as the reasoning.

        Args:
            response: Raw LLM response string.

        Returns:
            ``(is_correct, reasoning)`` tuple. Defaults to ``(False, response)``
            if the verdict line cannot be parsed.
        """
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        verdict_line = lines[0] if lines else ""
        reasoning = " ".join(lines[1:]) if len(lines) > 1 else ""

        match = re.search(
            r"VERDICT:\s*(correct|incorrect)", verdict_line, re.IGNORECASE
        )
        if match:
            is_correct = match.group(1).lower() == "correct"
            return is_correct, reasoning or verdict_line

        # Fallback: scan entire response for verdict keyword
        text_lower = response.lower()
        if "verdict: correct" in text_lower:
            return True, response
        if "verdict: incorrect" in text_lower:
            return False, response

        logger.warning(
            "LLMJudge could not parse verdict from response: %r", response[:120]
        )
        return False, response
