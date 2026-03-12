"""Base classes shared by all GraphBench RAG pipelines.

Defines:
- ``PipelineResult`` — structured output from a single pipeline answer() call.
- ``Pipeline`` — abstract base class that all concrete pipelines must implement.
- ``PROMPT_TEMPLATE`` — the single, shared LLM prompt used by both pipelines.

Both GraphRAG and GNN-RAG MUST use PROMPT_TEMPLATE unchanged to ensure
a fair comparison (same LLM instructions, same output format).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared prompt — NEVER modify per-pipeline; both pipelines use this exactly.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a precise question-answering assistant.
Use ONLY the provided context to answer the question.
If the context does not contain enough information, say "I don't know."
Keep your answer concise — one sentence or a short phrase is preferred.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Structured result returned by Pipeline.answer().

    Attributes:
        question: The original input question.
        predicted_answer: The pipeline's predicted answer string.
        gold_answer: Ground-truth answer (optional; populated by Evaluator).
        context_triples: List of (subject, relation, object) triples used as context.
        latency_ms: Wall-clock time for the answer() call in milliseconds.
        pipeline_name: Name of the pipeline that produced this result.
        metadata: Arbitrary extra info (community IDs, GNN scores, etc.).
    """

    question: str
    predicted_answer: str
    gold_answer: str | None = None
    context_triples: list[tuple[str, str, str]] = field(default_factory=list)
    latency_ms: float = 0.0
    pipeline_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class Pipeline(ABC):
    """Abstract base class for GraphBench RAG pipelines.

    Subclasses must implement:
    - ``answer(question)`` — retrieve context and generate an answer.
    - ``name`` — a human-readable pipeline identifier.

    Latency is measured by the caller (Evaluator) using time.perf_counter()
    around the answer() call ONLY. Do not measure latency inside answer().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pipeline name used in logs and results."""
        ...

    @abstractmethod
    def answer(self, question: str) -> PipelineResult:
        """Answer a single question using the pipeline's retrieval + LLM strategy.

        Args:
            question: A natural language question string.

        Returns:
            PipelineResult with predicted_answer and context_triples populated.
            latency_ms will be overwritten by the Evaluator.
        """
        ...

    def _check_clients(self, required: list[tuple[str, Any]]) -> None:
        """Raise RuntimeError if any required client attribute is None.

        Args:
            required: List of (name, attr) pairs to validate.

        Raises:
            RuntimeError: Lists all missing clients by name.
        """
        missing = [name for name, attr in required if attr is None]
        if missing:
            raise RuntimeError(
                f"{self.__class__.__name__}.answer() requires: {', '.join(missing)}. "
                "Pass them to the constructor."
            )

    def _empty_result(self, question: str, reason: str) -> "PipelineResult":
        """Return a no-context PipelineResult with 'I don't know.' answer.

        Args:
            question: The original question string.
            reason: Reason for the empty result (logged as a warning).

        Returns:
            PipelineResult with predicted_answer set to ``"I don't know."``.
        """
        logger.warning("%s returning empty result: %s", self.name, reason)
        return PipelineResult(
            question=question,
            predicted_answer="I don't know.",
            context_triples=[],
            pipeline_name=self.name,
            metadata={"reason": reason},
        )

    def build_prompt(self, question: str, triples: list[tuple[str, str, str]]) -> str:
        """Format the shared PROMPT_TEMPLATE with retrieved triples as context.

        Args:
            question: Natural language question.
            triples: List of (subject, relation, object) triples.

        Returns:
            Formatted prompt string ready for the LLM.
        """
        context_lines = [f"{s} {r} {o}." for s, r, o in triples]
        context = "\n".join(context_lines) if context_lines else "No context available."
        return PROMPT_TEMPLATE.format(context=context, question=question)
