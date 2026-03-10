"""PyTorch Geometric dataset builder for GNN link-prediction training.

KGDataset converts KG triples + pre-computed entity embeddings into PyG
Data objects with:
- Separate edge_index (message-passing graph, training positives only) and
  edge_label_index (supervision edges) to prevent data leakage.
- Vectorised negative sampling (corrupt head OR tail, 1 negative per positive).
- Deterministic 80/10/10 train/val/test split (seed=42 by default).

Critical design: val/test splits share the training-positive edge_index for
message passing. Their own val/test positive edges are EXCLUDED from the
message-passing graph to prevent the model from attending over edges it is
asked to predict.
"""

import logging

import numpy as np
import torch
from torch_geometric.data import Data

from graphbench.ingestion import Triple

logger = logging.getLogger(__name__)


class KGDataset:
    """Builds PyG train/val/test Data objects from KG triples and embeddings.

    Args:
        triples: List of Triple TypedDicts (subject, relation, object).
        entity_embeddings: Dict mapping entity string → 384-dim numpy array.
            Entities absent from this dict receive zero-vector features.
        neg_sampling_ratio: Number of negatives per positive edge (default 1).
        seed: Random seed for negative sampling and splits (default 42).
        embedding_dim: Node feature dimensionality (default 384).
    """

    def __init__(
        self,
        triples: list[Triple],
        entity_embeddings: dict[str, np.ndarray],
        neg_sampling_ratio: int = 1,
        seed: int = 42,
        embedding_dim: int = 384,
    ) -> None:
        self.triples = triples
        self.entity_embeddings = entity_embeddings
        self.neg_sampling_ratio = neg_sampling_ratio
        self.seed = seed
        self.embedding_dim = embedding_dim

        # Build sorted entity list and index map once at construction
        self.entities: list[str] = sorted(
            {t["subject"] for t in triples} | {t["object"] for t in triples}
        )
        self.node_index: dict[str, int] = {e: i for i, e in enumerate(self.entities)}
        self.n_nodes = len(self.entities)

        logger.info(
            "KGDataset: %d triples, %d unique entities.", len(triples), self.n_nodes
        )

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------

    def _build_node_features(self) -> torch.Tensor:
        """Build node feature matrix from pre-computed embeddings.

        Entities absent from entity_embeddings receive zero vectors.

        Returns:
            Float32 tensor of shape (n_nodes, embedding_dim).
        """
        rows = []
        missing = 0
        for entity in self.entities:
            vec = self.entity_embeddings.get(entity)
            if vec is None:
                missing += 1
                vec = np.zeros(self.embedding_dim, dtype=np.float32)
            rows.append(vec.astype(np.float32))

        if missing:
            logger.warning(
                "%d / %d entities not found in embedding index; using zero vectors.",
                missing,
                self.n_nodes,
            )

        return torch.tensor(np.stack(rows), dtype=torch.float)

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _sample_negatives(
        self,
        pos_src: np.ndarray,
        pos_dst: np.ndarray,
        pos_edge_set: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample negative edges by corrupting head or tail uniformly.

        For each positive edge, corrupt either the head (src) or tail (dst)
        with 50/50 probability. Resulting negatives that accidentally match a
        known positive are re-sampled (max 3 retries per edge).

        Args:
            pos_src: Source node indices of positive edges.
            pos_dst: Destination node indices of positive edges.
            pos_edge_set: Set of all positive (src, dst) pairs for filtering.
            rng: NumPy random Generator instance.

        Returns:
            Tuple of (neg_src, neg_dst) numpy int64 arrays.
        """
        n = len(pos_src)
        neg_src = pos_src.copy()
        neg_dst = pos_dst.copy()

        corrupt_head = rng.integers(0, 2, size=n).astype(bool)
        neg_src[corrupt_head] = rng.integers(0, self.n_nodes, size=corrupt_head.sum())
        neg_dst[~corrupt_head] = rng.integers(
            0, self.n_nodes, size=(~corrupt_head).sum()
        )

        # Re-sample any accidental positives (up to 3 attempts)
        for _ in range(3):
            mask = np.array(
                [(int(neg_src[i]), int(neg_dst[i])) in pos_edge_set for i in range(n)]
            )
            if not mask.any():
                break
            n_bad = mask.sum()
            new_src = rng.integers(0, self.n_nodes, size=n_bad)
            new_dst = rng.integers(0, self.n_nodes, size=n_bad)
            neg_src[mask] = new_src
            neg_dst[mask] = new_dst

        return neg_src.astype(np.int64), neg_dst.astype(np.int64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> tuple[Data, Data, Data]:
        """Build and split the dataset into train / val / test PyG Data objects.

        All three Data objects share the same node feature matrix x. The
        edge_index (message-passing graph) in all three contains only
        training-positive edges to prevent leakage.

        Args:
            train_ratio: Fraction of positive edges for training (default 0.8).
            val_ratio: Fraction for validation (default 0.1). Remainder → test.

        Returns:
            Tuple of (train_data, val_data, test_data) PyG Data objects, each with:
            - x: Node features (N, 384), shared.
            - edge_index: Training-positive message-passing graph (2, E_train).
            - edge_label_index: Supervision edges, pos+neg (2, E_sup).
            - edge_label: Binary labels for supervision edges (E_sup,).
        """
        rng = np.random.default_rng(self.seed)
        x = self._build_node_features()

        # Build all positive edges
        all_src = np.array([self.node_index[t["subject"]] for t in self.triples])
        all_dst = np.array([self.node_index[t["object"]] for t in self.triples])
        n_pos = len(all_src)

        # Shuffle indices
        perm = rng.permutation(n_pos)
        all_src, all_dst = all_src[perm], all_dst[perm]

        # Split boundary indices
        n_train = int(n_pos * train_ratio)
        n_val = int(n_pos * val_ratio)

        train_src, train_dst = all_src[:n_train], all_dst[:n_train]
        val_src, val_dst = (
            all_src[n_train : n_train + n_val],
            all_dst[n_train : n_train + n_val],
        )
        test_src, test_dst = all_src[n_train + n_val :], all_dst[n_train + n_val :]

        # Message-passing graph: only training-positive edges
        train_edge_index = torch.from_numpy(
            np.stack([train_src, train_dst], axis=0)
        ).long()

        # Full positive edge set (for negative filtering across ALL splits)
        all_pos_set: set[tuple[int, int]] = {
            (int(s), int(d)) for s, d in zip(all_src, all_dst)
        }

        logger.info(
            "Split: %d train / %d val / %d test positive edges.",
            len(train_src),
            len(val_src),
            len(test_src),
        )

        def _build_split_data(pos_src: np.ndarray, pos_dst: np.ndarray) -> Data:
            neg_src, neg_dst = self._sample_negatives(
                pos_src, pos_dst, all_pos_set, rng
            )
            # Interleave positives then negatives
            sup_src = np.concatenate([pos_src, neg_src])
            sup_dst = np.concatenate([pos_dst, neg_dst])
            labels = np.concatenate([np.ones(len(pos_src)), np.zeros(len(neg_src))])
            return Data(
                x=x,
                edge_index=train_edge_index,
                edge_label_index=torch.tensor([sup_src, sup_dst], dtype=torch.long),
                edge_label=torch.tensor(labels, dtype=torch.float),
                num_nodes=self.n_nodes,
            )

        train_data = _build_split_data(train_src, train_dst)
        val_data = _build_split_data(val_src, val_dst)
        test_data = _build_split_data(test_src, test_dst)

        return train_data, val_data, test_data
