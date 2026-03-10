"""Louvain community detection on knowledge graph subgraphs.

Runs the Louvain algorithm (python-louvain / ``community`` library) on a
NetworkX undirected graph extracted from Neo4j, with ``resolution=0.8``.

Returns a mapping ``{entity: community_id}`` and groups triples by community.

The top-K communities most relevant to the input question are selected by
counting the overlap between each community's entities and the seed entities
returned by FAISS search for the question embedding. This forms the context
for GraphRAG's LLM prompt.

Used by GraphRAGPipeline. Import as::

    from graphbench.community.detector import CommunityDetector
"""

from __future__ import annotations

import logging
from collections import defaultdict

from graphbench.utils.config import settings

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detect and rank Louvain communities in a KG subgraph.

    Args:
        resolution: Louvain resolution parameter (higher → more communities).
            Defaults to ``settings.community_resolution`` (0.8).
        random_state: Random seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        resolution: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.resolution = (
            resolution if resolution is not None else settings.community_resolution
        )
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, graph) -> dict[str, int]:
        """Run Louvain community detection on an undirected NetworkX graph.

        If a directed graph (DiGraph) is passed it is converted to undirected
        before running Louvain (python-louvain requires undirected input).

        Args:
            graph: A ``networkx.Graph`` or ``networkx.DiGraph`` instance.

        Returns:
            ``{entity_string: community_id}`` partition dictionary.
            Returns an empty dict for an empty or single-node graph.
        """
        import community as community_louvain  # noqa: PLC0415 — python-louvain
        import networkx as nx  # noqa: PLC0415

        if isinstance(graph, nx.DiGraph):
            graph = graph.to_undirected()

        if graph.number_of_nodes() == 0:
            logger.warning("Empty graph passed to CommunityDetector — no communities.")
            return {}

        partition: dict[str, int] = community_louvain.best_partition(
            graph,
            resolution=self.resolution,
            random_state=self.random_state,
        )
        n_communities = len(set(partition.values()))
        logger.debug(
            "Louvain detected %d communities over %d nodes (resolution=%.2f).",
            n_communities,
            graph.number_of_nodes(),
            self.resolution,
        )
        return partition

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def group_triples(
        self,
        partition: dict[str, int],
        triples: list[tuple[str, str, str]],
    ) -> dict[int, list[tuple[str, str, str]]]:
        """Group triples by community ID.

        A triple is assigned to the community of its *subject* entity.
        Triples whose subject is not in the partition are placed in
        community ``-1`` (unassigned).

        Args:
            partition: ``{entity: community_id}`` mapping from :meth:`detect`.
            triples: List of ``(subject, relation, object)`` triples.

        Returns:
            ``{community_id: [triple, ...]}`` grouping.
        """
        groups: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
        for subj, rel, obj in triples:
            cid = partition.get(subj, partition.get(obj, -1))
            groups[cid].append((subj, rel, obj))
        return dict(groups)

    # ------------------------------------------------------------------
    # Community selection
    # ------------------------------------------------------------------

    def select_top_communities(
        self,
        groups: dict[int, list[tuple[str, str, str]]],
        seed_entities: list[str],
        k: int = 3,
    ) -> list[int]:
        """Select the top-k communities most relevant to the query.

        Relevance is measured by the overlap between each community's
        entities and the FAISS-retrieved seed entities for the question.
        Communities with higher entity overlap are preferred; ties are
        broken by community size (most triples first).

        Args:
            groups: ``{community_id: [triple, ...]}`` from :meth:`group_triples`.
            seed_entities: Entity strings retrieved by FAISS for the question.
            k: Number of communities to select (default: 3).

        Returns:
            List of up to ``k`` community IDs, ordered by descending relevance.
        """
        if not groups:
            return []

        seed_set = set(e.lower() for e in seed_entities)

        def score(cid: int) -> tuple[int, int]:
            triples = groups[cid]
            # Entities in this community
            entities = set()
            for subj, _, obj in triples:
                entities.add(subj.lower())
                entities.add(obj.lower())
            overlap = len(entities & seed_set)
            size = len(triples)
            return (overlap, size)  # sort descending on both

        ranked = sorted(groups.keys(), key=score, reverse=True)
        selected = ranked[:k]
        logger.debug(
            "Selected %d communities from %d total (seed_entities=%d).",
            len(selected),
            len(groups),
            len(seed_entities),
        )
        return selected
