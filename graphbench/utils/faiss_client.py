"""FAISS vector index client for GraphBench.

Manages a 384-dimensional IndexFlatIP (inner-product) FAISS index over
entity embeddings. Inner product on L2-normalised vectors equals cosine
similarity. Used exclusively for vector search — Neo4j handles graph traversal.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from graphbench.utils.config import settings

if TYPE_CHECKING:
    import faiss as faiss_type

logger = logging.getLogger(__name__)


class FAISSClient:
    """FAISS IndexFlatIP client for entity cosine-similarity search.

    Usage:
        # Load from disk (production)
        client = FAISSClient.load()
        results = client.search(query_embedding, k=10)

        # Build in-memory (testing / one-off)
        client = FAISSClient.build(entity_strings, embeddings)
    """

    def __init__(
        self,
        index: "faiss_type.IndexFlatIP",
        id_map: dict[int, str],
    ) -> None:
        """Initialise with a pre-built index and integer-to-entity mapping.

        Args:
            index: A built faiss.IndexFlatIP.
            id_map: Mapping from integer FAISS ID → entity surface string.
        """
        self._index = index
        self._id_map = id_map
        logger.info(
            "FAISSClient ready: %d vectors, dim=%d.",
            index.ntotal,
            index.d,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, index_path: Path | None = None) -> "FAISSClient":
        """Load a persisted FAISS index and id_map from disk.

        Expects two files:
        - {index_path}.faiss
        - {index_path}_id_map.json

        Args:
            index_path: Path prefix (without .faiss extension).
                Defaults to settings.faiss_index_path.

        Returns:
            Initialised FAISSClient instance.

        Raises:
            FileNotFoundError: If either expected file is missing.
        """
        resolved = Path(index_path or settings.faiss_index_path)
        faiss_file = resolved.with_suffix(".faiss")
        id_map_file = resolved.parent / (resolved.stem + "_id_map.json")

        if not faiss_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_file}")
        if not id_map_file.exists():
            raise FileNotFoundError(f"id_map not found: {id_map_file}")

        import faiss  # noqa: PLC0415 — lazy to avoid Mac FAISS+torch conflict

        index = faiss.read_index(str(faiss_file))
        with id_map_file.open("r", encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)
        id_map = {int(k): v for k, v in raw.items()}

        logger.info(
            "Loaded FAISS index from %s (%d vectors).", faiss_file, index.ntotal
        )
        return cls(index, id_map)

    @classmethod
    def build(
        cls,
        entity_strings: list[str],
        embeddings: np.ndarray,
    ) -> "FAISSClient":
        """Build a FAISSClient in memory from entity strings and embeddings.

        Does not persist to disk. Use this for testing or ephemeral usage.

        Args:
            entity_strings: List of entity surface form strings.
            embeddings: float32 numpy array, shape (n, dim), L2-normalised.

        Returns:
            Initialised FAISSClient instance.
        """
        import faiss  # noqa: PLC0415 — lazy to avoid Mac FAISS+torch conflict

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        id_map = {i: s for i, s in enumerate(entity_strings)}
        return cls(index, id_map)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Find the top-k most similar entities to a query embedding.

        Args:
            query_embedding: float32 array of shape (384,) or (1, 384),
                L2-normalised for cosine similarity results.
            k: Number of results to return. Defaults to settings.top_k_faiss.

        Returns:
            List of (entity_string, score) tuples, sorted descending by score.
            Score is cosine similarity in [-1.0, 1.0].
        """
        resolved_k = k if k is not None else settings.top_k_faiss

        vec = np.array(query_embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        scores, indices = self._index.search(vec, resolved_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
            entity = self._id_map.get(int(idx), f"<unknown:{idx}>")
            results.append((entity, float(score)))

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal
