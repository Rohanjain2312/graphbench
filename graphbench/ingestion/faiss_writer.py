"""FAISS index writer for the ingestion pipeline.

Takes entity embeddings produced by embedder.py and builds a 384-dim
IndexFlatIP FAISS index. Persists two files:
- {path}.faiss          — binary FAISS index
- {path}_id_map.json    — {str(int_id): entity_string} mapping for result resolution
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from graphbench.utils.config import settings

logger = logging.getLogger(__name__)


def _validate_inputs(entity_strings: list[str], embeddings: np.ndarray) -> None:
    """Validate shape, dtype, and length consistency of inputs.

    Args:
        entity_strings: List of entity surface form strings.
        embeddings: float32 numpy array of shape (n, embedding_dim).

    Raises:
        ValueError: On any shape, dtype, or length mismatch.
    """
    if len(entity_strings) != embeddings.shape[0]:
        raise ValueError(
            f"entity_strings length ({len(entity_strings)}) != "
            f"embeddings rows ({embeddings.shape[0]})"
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != settings.embedding_dim:
        raise ValueError(
            f"Expected embeddings shape (n, {settings.embedding_dim}), "
            f"got {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        raise ValueError(f"Embeddings must be float32, got {embeddings.dtype}")


def build_and_save_index(
    entity_strings: list[str],
    embeddings: np.ndarray,
    *,
    index_path: Path | None = None,
) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP from embeddings and persist to disk.

    Saves two files:
    - {index_path}.faiss         — binary FAISS index
    - {index_path}_id_map.json   — integer-ID → entity-string mapping

    Args:
        entity_strings: Entity surface forms, parallel to embeddings rows.
        embeddings: float32 array of shape (n, 384), L2-normalised.
        index_path: Path prefix (no extension). Defaults to settings.faiss_index_path.

    Returns:
        The built faiss.IndexFlatIP (also persisted to disk).

    Raises:
        ValueError: If inputs are inconsistent or incorrectly shaped.
    """
    _validate_inputs(entity_strings, embeddings)

    resolved = Path(index_path or settings.faiss_index_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(settings.embedding_dim)
    index.add(embeddings)
    logger.info(
        "Built FAISS IndexFlatIP with %d vectors (dim=%d).",
        index.ntotal,
        settings.embedding_dim,
    )

    # Persist binary index
    faiss_file = resolved.with_suffix(".faiss")
    faiss.write_index(index, str(faiss_file))
    logger.info("Saved FAISS index → %s", faiss_file)

    # Persist id_map: JSON requires string keys
    id_map = {str(i): entity for i, entity in enumerate(entity_strings)}
    id_map_file = resolved.parent / (resolved.stem + "_id_map.json")
    with id_map_file.open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=None)
    logger.info("Saved id_map (%d entries) → %s", len(id_map), id_map_file)

    return index
