"""Community summarizer for GraphRAG context preparation.

Converts detected communities (sets of triples) into an ordered list of
context triples suitable for inclusion in the LLM prompt via
``Pipeline.build_prompt()``.

Strategies:

- **ranked** (default): sort triples by entity frequency so that the most
  central entities appear first. More central entities are more likely to
  be relevant to the question.
- **simple**: concatenate triples in original order (no ranking).

Both produce a ``list[tuple[str, str, str]]`` so they are interchangeable
and can be used for ablation studies.

Used by GraphRAGPipeline::

    from graphbench.community.summarizer import merge_community_triples
"""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)


def merge_community_triples(
    groups: dict[int, list[tuple[str, str, str]]],
    selected_ids: list[int],
    max_triples: int = 100,
    ranked: bool = True,
) -> list[tuple[str, str, str]]:
    """Merge and optionally rank triples from selected communities.

    Combines triples from the chosen community IDs into a single ordered
    list. If ``ranked=True``, triples are sorted so that entities appearing
    most frequently across all selected communities come first.

    Args:
        groups: ``{community_id: [triple, ...]}`` mapping from
            :class:`~graphbench.community.detector.CommunityDetector`.
        selected_ids: Community IDs to include (in preference order).
        max_triples: Hard cap on the number of returned triples (default: 100).
        ranked: If True, rank by entity centrality; otherwise preserve
            insertion order (community order, then original triple order).

    Returns:
        Ordered list of ``(subject, relation, object)`` triples, length
        ``≤ max_triples``.
    """
    merged: list[tuple[str, str, str]] = []
    for cid in selected_ids:
        merged.extend(groups.get(cid, []))

    if not merged:
        logger.debug(
            "merge_community_triples: no triples in selected communities %s.",
            selected_ids,
        )
        return []

    if ranked:
        merged = _rank_by_entity_frequency(merged)

    result = merged[:max_triples]
    logger.debug(
        "merge_community_triples: %d triples from communities %s (ranked=%s, max=%d).",
        len(result),
        selected_ids,
        ranked,
        max_triples,
    )
    return result


def _rank_by_entity_frequency(
    triples: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Sort triples so the most central entities appear first.

    Centrality is approximated by raw entity frequency across all triples
    (subject + object positions). A triple's score is the sum of its
    subject and object frequencies.

    Args:
        triples: Unordered list of ``(subject, relation, object)`` triples.

    Returns:
        Same triples sorted by descending entity-centrality score.
    """
    freq: Counter[str] = Counter()
    for subj, _, obj in triples:
        freq[subj] += 1
        freq[obj] += 1

    return sorted(triples, key=lambda t: freq[t[0]] + freq[t[2]], reverse=True)
