"""Data ingestion pipeline: REBEL extraction → embeddings → FAISS + Neo4j."""

from typing import TypedDict


class Triple(TypedDict):
    """A single knowledge graph triple.

    Attributes:
        subject: Subject entity surface form string.
        relation: Relation type in snake_case.
        object: Object entity surface form string.
    """

    subject: str
    relation: str
    object: str


__all__ = ["Triple"]
