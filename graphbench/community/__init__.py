"""Community detection module: Louvain algorithm and community summarization."""

from graphbench.community.detector import CommunityDetector
from graphbench.community.summarizer import merge_community_triples

__all__ = ["CommunityDetector", "merge_community_triples"]
