"""Evaluation metrics for GraphBench: Exact Match and token-level F1.

normalize_answer() is the canonical normalisation function used across
the entire codebase. NEVER reimplement it elsewhere — always import from here.

Follows the official HotpotQA evaluation script:
- Lowercase
- Remove punctuation
- Remove articles (a, an, the)
- Collapse whitespace

Both EM and F1 are computed over the normalised forms.

Implementation: Phase 5 (benchmark) — but this file is complete in Phase 1
so other modules can safely import normalize_answer() from the start.
"""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalise an answer string for EM/F1 evaluation.

    Applies lowercasing, punctuation removal, article removal, and
    whitespace normalisation. This is the single canonical normaliser
    for the entire GraphBench codebase.

    Args:
        text: Raw answer string (predicted or gold).

    Returns:
        Normalised answer string.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(predicted: str, gold: str) -> float:
    """Compute Exact Match score between predicted and gold answers.

    Args:
        predicted: Predicted answer string.
        gold: Gold (ground-truth) answer string.

    Returns:
        1.0 if normalised strings match exactly, 0.0 otherwise.
    """
    return float(normalize_answer(predicted) == normalize_answer(gold))


def token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 score between predicted and gold answers.

    Args:
        predicted: Predicted answer string.
        gold: Gold (ground-truth) answer string.

    Returns:
        F1 score in [0.0, 1.0].
    """
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
