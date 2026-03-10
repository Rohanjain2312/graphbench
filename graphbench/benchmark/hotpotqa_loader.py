"""HotpotQA distractor-setting dataset loader for the benchmark.

Loads 500 questions from HotpotQA (HuggingFace: ``hotpot_qa``, distractor config):
- 250 bridge questions (require multi-hop reasoning)
- 250 comparison questions (require comparing two entities)

The 500-question subset is deterministically sampled (seed=42) so results are
reproducible across machines and runs.

Each returned dict has keys:
- ``id``: original HotpotQA question ID
- ``question``: question string
- ``answer``: gold answer string
- ``type``: ``"bridge"`` or ``"comparison"``

Usage::

    from graphbench.benchmark.hotpotqa_loader import load_hotpotqa

    questions = load_hotpotqa()          # 500 questions, seed=42
    questions = load_hotpotqa(n=100)     # 100 questions (50+50), seed=42
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace dataset identifier
_HF_DATASET = "hotpot_qa"
_HF_CONFIG = "distractor"
_HF_SPLIT = "train"  # validation split is too small; use train


def load_hotpotqa(
    n: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Load a balanced subset of HotpotQA distractor questions.

    Samples ``n // 2`` bridge questions and ``n // 2`` comparison questions
    deterministically from the HotpotQA training split.

    Args:
        n: Total number of questions to return (must be even). Default: 500.
        seed: Random seed for reproducibility. Default: 42.

    Returns:
        List of question dicts with keys ``id``, ``question``, ``answer``,
        ``type``. Length is exactly ``n`` (or less if the dataset has fewer
        examples of a given type).

    Raises:
        ValueError: If ``n`` is not a positive even integer.
        ImportError: If the ``datasets`` package is not installed.
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"n must be a positive even integer, got {n}.")

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for load_hotpotqa(). "
            "Install it with: pip install datasets"
        ) from exc

    logger.info(
        "Loading HotpotQA (%s/%s/%s) — sampling %d questions (seed=%d).",
        _HF_DATASET,
        _HF_CONFIG,
        _HF_SPLIT,
        n,
        seed,
    )

    dataset = load_dataset(
        _HF_DATASET, _HF_CONFIG, split=_HF_SPLIT, trust_remote_code=True
    )

    bridge = [_to_dict(row) for row in dataset if row["type"] == "bridge"]
    comparison = [_to_dict(row) for row in dataset if row["type"] == "comparison"]

    logger.info(
        "HotpotQA pool: %d bridge, %d comparison.", len(bridge), len(comparison)
    )

    rng = np.random.default_rng(seed)
    n_each = n // 2

    sampled_bridge = _sample(bridge, n_each, rng)
    sampled_comparison = _sample(comparison, n_each, rng)

    questions = sampled_bridge + sampled_comparison
    logger.info(
        "Returning %d questions (%d bridge + %d comparison).",
        len(questions),
        len(sampled_bridge),
        len(sampled_comparison),
    )
    return questions


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _to_dict(row: dict) -> dict:
    """Extract the fields we need from a HotpotQA dataset row."""
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
    }


def _sample(items: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    """Sample up to ``k`` items without replacement using ``rng``."""
    if len(items) <= k:
        return list(items)
    indices = rng.choice(len(items), size=k, replace=False)
    return [items[int(i)] for i in indices]
