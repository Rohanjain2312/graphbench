"""Entity and relation text embedder using sentence-transformers.

Encodes entity surface forms into 384-dimensional L2-normalised vectors
using all-MiniLM-L6-v2. Produces float32 numpy arrays for FAISS IndexFlatIP.

L2-normalised embeddings enable cosine similarity via inner product on
IndexFlatIP — no need for IndexFlatL2 or post-hoc normalisation.
"""

import logging

import numpy as np

from graphbench.utils.config import settings

logger = logging.getLogger(__name__)


def embed_entities(
    entity_strings: list[str],
    *,
    model_name: str | None = None,
    batch_size: int = 256,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode entity surface forms into 384-dim L2-normalised embeddings.

    Uses sentence-transformers/all-MiniLM-L6-v2 by default. Returns float32
    numpy array suitable for FAISS IndexFlatIP (inner product on L2-normalised
    vectors equals cosine similarity).

    Args:
        entity_strings: List of entity surface form strings to encode.
        model_name: Override the model name. Defaults to settings.embedding_model.
        batch_size: Number of strings per encoding batch (256 fits ~2GB RAM).
        show_progress: Whether to display a tqdm progress bar during encoding.

    Returns:
        np.ndarray of shape (len(entity_strings), 384), dtype float32,
        with each row L2-normalised to unit length.

    Raises:
        ValueError: If entity_strings is empty.
    """
    if not entity_strings:
        raise ValueError("entity_strings must not be empty.")

    # Lazy import: sentence_transformers pulls in torch, which is heavy and
    # GPU-dependent. Deferring the import prevents crashes in environments
    # where torch is not properly initialised (e.g. CI without GPU).
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    resolved_model = model_name or settings.embedding_model
    logger.info(
        "Loading embedding model '%s' for %d entities.",
        resolved_model,
        len(entity_strings),
    )

    model = SentenceTransformer(resolved_model)

    embeddings = model.encode(
        entity_strings,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalise in-place (cosine via IP)
        convert_to_numpy=True,
    )

    result = embeddings.astype(np.float32)
    logger.info(
        "Encoded %d entities → shape %s, dtype %s.",
        len(entity_strings),
        result.shape,
        result.dtype,
    )
    return result
