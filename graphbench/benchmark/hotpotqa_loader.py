"""HotpotQA distractor-setting dataset loader for the benchmark.

Loads 500 questions from HotpotQA (HuggingFace: hotpot_qa, distractor config):
- 250 bridge questions (require multi-hop reasoning)
- 250 comparison questions (require comparing two entities)

Returns a list of dicts with keys: id, question, answer, type, supporting_facts.
The 500-question subset is deterministically sampled (seed=42) to ensure
reproducibility across runs and systems.

Implementation: Phase 5 (benchmark).
"""
