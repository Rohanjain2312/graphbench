"""Community summarizer for GraphRAG context preparation.

Converts a detected community (a set of triples) into a concise text
context string suitable for inclusion in the LLM prompt.

Strategies:
- Simple: concatenate all triples as "S relation O." sentences.
- Ranked: sort triples by entity frequency (most central entities first).

The ranked strategy is the default for GraphRAG. Both produce the same
format so they are interchangeable for ablation studies.

Implementation: Phase 4 (pipelines) — used by GraphRAGPipeline.
"""
