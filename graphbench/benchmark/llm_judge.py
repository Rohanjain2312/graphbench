"""LLM-based answer quality judge for qualitative benchmark analysis.

Uses a lightweight LLM (Phi-3-mini via Ollama locally, or Mistral-7B on Colab)
to judge whether a predicted answer is semantically correct given the gold answer,
even when token-level F1 fails (e.g., paraphrases, date formats, aliases).

Output: binary correct/incorrect + brief reasoning string per question.
Used for qualitative analysis only — EM and F1 remain the primary metrics.

Implementation: Phase 5 (benchmark).
"""
