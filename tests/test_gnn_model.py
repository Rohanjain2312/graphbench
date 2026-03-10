"""Tests for graphbench.gnn.model (GATModel).

All tests run on CPU without GPU. Synthetic tiny graphs are used —
no KG data or model downloads required.
"""

import numpy as np
import pytest


@pytest.fixture
def tiny_graph():
    """Synthetic 10-node, 20-edge CPU graph for GATModel tests."""
    # Lazy imports: torch + torch_geometric must not be loaded at collection time
    # on Mac — they conflict with FAISS when both are in the same process.
    import torch
    from torch_geometric.data import Data

    torch.manual_seed(0)
    n_nodes, n_edges, feat_dim = 10, 20, 384
    x = torch.randn(n_nodes, feat_dim)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_label_index = edge_index.clone()
    return Data(x=x, edge_index=edge_index, edge_label_index=edge_label_index)


# ---------------------------------------------------------------------------
# Core model tests (previously skipped — now fully implemented)
# ---------------------------------------------------------------------------


def test_gat_model_layer_count() -> None:
    """GATModel should have exactly 3 GATConv layers in model.convs."""
    from graphbench.gnn.model import GATModel

    model = GATModel()
    assert len(model.convs) == 3, f"Expected 3 GATConv layers, got {len(model.convs)}"


def test_gat_model_attention_heads() -> None:
    """Every GATConv layer should have exactly 4 attention heads."""
    from graphbench.gnn.model import GATModel

    model = GATModel()
    for i, conv in enumerate(model.convs):
        assert conv.heads == 4, f"Layer {i}: expected 4 heads, got {conv.heads}"


def test_gat_model_forward_pass(tiny_graph) -> None:
    """Forward pass should return a 1-D logit tensor with one entry per candidate edge."""
    import torch

    from graphbench.gnn.model import GATModel

    model = GATModel()
    model.eval()
    with torch.no_grad():
        logits = model(tiny_graph.x, tiny_graph.edge_index, tiny_graph.edge_label_index)

    expected_len = tiny_graph.edge_label_index.shape[1]
    assert logits.shape == (
        expected_len,
    ), f"Expected shape ({expected_len},), got {logits.shape}"
    assert logits.dtype == torch.float32


def test_gat_model_output_in_0_1(tiny_graph) -> None:
    """Sigmoid-normalised edge scores should all lie in [0, 1]."""
    import torch

    from graphbench.gnn.model import GATModel

    model = GATModel()
    model.eval()
    with torch.no_grad():
        logits = model(tiny_graph.x, tiny_graph.edge_index, tiny_graph.edge_label_index)
        scores = torch.sigmoid(logits)

    assert scores.min().item() >= 0.0 - 1e-6
    assert scores.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# encode / decode separation
# ---------------------------------------------------------------------------


def test_encode_output_shape(tiny_graph) -> None:
    """encode() should return node embeddings of shape (N, out_channels)."""
    import torch

    from graphbench.gnn.model import GATModel

    model = GATModel(out_channels=32)
    model.eval()
    with torch.no_grad():
        z = model.encode(tiny_graph.x, tiny_graph.edge_index)

    assert z.shape == (tiny_graph.x.shape[0], 32)


def test_decode_output_shape(tiny_graph) -> None:
    """decode() should return one logit per candidate edge."""
    import torch

    from graphbench.gnn.model import GATModel

    model = GATModel()
    model.eval()
    with torch.no_grad():
        z = model.encode(tiny_graph.x, tiny_graph.edge_index)
        logits = model.decode(z, tiny_graph.edge_label_index)

    assert logits.shape == (tiny_graph.edge_label_index.shape[1],)


def test_score_edges_returns_probabilities(tiny_graph) -> None:
    """score_edges() should return values in [0, 1] (sigmoid applied)."""
    from graphbench.gnn.model import GATModel

    model = GATModel()
    scores = model.score_edges(
        tiny_graph.x, tiny_graph.edge_index, tiny_graph.edge_label_index
    )
    assert scores.min().item() >= 0.0 - 1e-6
    assert scores.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Checkpoint round-trip test
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load(tmp_path, tiny_graph) -> None:
    """Saving then loading a checkpoint should restore identical model weights."""
    import torch

    from graphbench.gnn.model import GATModel
    from graphbench.utils.checkpoint import load_best_checkpoint, save_checkpoint

    model = GATModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_checkpoint(model, optimizer, epoch=1, val_auc=0.80, checkpoint_dir=tmp_path)

    checkpoint, _ = load_best_checkpoint(tmp_path)
    assert checkpoint["epoch"] == 1
    assert abs(checkpoint["val_auc"] - 0.80) < 1e-6

    model2 = GATModel()
    model2.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model(tiny_graph.x, tiny_graph.edge_index, tiny_graph.edge_label_index)
        out2 = model2(tiny_graph.x, tiny_graph.edge_index, tiny_graph.edge_label_index)

    assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# KGDataset tests
# ---------------------------------------------------------------------------


class TestKGDataset:
    """Tests for graphbench.gnn.dataset.KGDataset."""

    @pytest.fixture
    def sample_triples_and_embeddings(
        self,
    ) -> tuple[list, dict]:
        """Return tiny Triple list + embedding dict for dataset tests."""
        from graphbench.ingestion import Triple

        triples: list[Triple] = [
            {"subject": "alice", "relation": "knows", "object": "bob"},
            {"subject": "bob", "relation": "knows", "object": "carol"},
            {"subject": "carol", "relation": "knows", "object": "dave"},
            {"subject": "dave", "relation": "knows", "object": "alice"},
            {"subject": "alice", "relation": "knows", "object": "carol"},
        ]
        rng = np.random.default_rng(0)
        entities = ["alice", "bob", "carol", "dave"]
        embs = {e: (rng.standard_normal(384).astype(np.float32)) for e in entities}
        # L2-normalise
        embs = {k: v / np.linalg.norm(v) for k, v in embs.items()}
        return triples, embs

    def test_split_returns_three_data_objects(
        self, sample_triples_and_embeddings
    ) -> None:
        """split() should return exactly three Data objects."""
        from graphbench.gnn.dataset import KGDataset

        triples, embs = sample_triples_and_embeddings
        ds = KGDataset(triples, embs)
        result = ds.split()
        assert len(result) == 3

    def test_x_shared_across_splits(self, sample_triples_and_embeddings) -> None:
        """All three splits should have the same x tensor (same data_ptr)."""
        from graphbench.gnn.dataset import KGDataset

        triples, embs = sample_triples_and_embeddings
        ds = KGDataset(triples, embs)
        train, val, test = ds.split()
        assert train.x.data_ptr() == val.x.data_ptr() == test.x.data_ptr()

    def test_edge_label_has_pos_and_neg(self, sample_triples_and_embeddings) -> None:
        """edge_label should contain both 1s (positive) and 0s (negative)."""
        from graphbench.gnn.dataset import KGDataset

        triples, embs = sample_triples_and_embeddings
        ds = KGDataset(triples, embs)
        train, _, _ = ds.split()
        assert (train.edge_label == 1).any()
        assert (train.edge_label == 0).any()

    def test_node_index_covers_all_entities(
        self, sample_triples_and_embeddings
    ) -> None:
        """KGDataset.node_index should map every unique entity to a unique int."""
        from graphbench.gnn.dataset import KGDataset

        triples, embs = sample_triples_and_embeddings
        ds = KGDataset(triples, embs)
        expected_entities = {"alice", "bob", "carol", "dave"}
        assert set(ds.node_index.keys()) == expected_entities
        # All IDs must be unique
        assert len(set(ds.node_index.values())) == len(ds.node_index)

    def test_zero_fallback_for_missing_embeddings(self) -> None:
        """Entities without embeddings should get zero-vector node features."""
        from graphbench.gnn.dataset import KGDataset

        triples = [{"subject": "x", "relation": "knows", "object": "y"}]
        embs: dict = {}  # no embeddings at all
        ds = KGDataset(triples, embs)
        train, _, _ = ds.split()
        import torch

        assert torch.all(train.x == 0)


# ---------------------------------------------------------------------------
# subgraph_to_pyg tests
# ---------------------------------------------------------------------------


class TestSubgraphToPyg:
    """Tests for graphbench.gnn.subgraph.subgraph_to_pyg."""

    @pytest.fixture
    def mini_triples(self):
        return [
            ("alice", "KNOWS", "bob"),
            ("bob", "KNOWS", "carol"),
        ]

    @pytest.fixture
    def mini_embeddings(self):
        rng = np.random.default_rng(1)
        return {
            "alice": rng.standard_normal(384).astype(np.float32),
            "bob": rng.standard_normal(384).astype(np.float32),
            "carol": rng.standard_normal(384).astype(np.float32),
        }

    def test_correct_node_count(self, mini_triples, mini_embeddings) -> None:
        """x should have one row per unique entity."""
        from graphbench.gnn.subgraph import subgraph_to_pyg

        data = subgraph_to_pyg(mini_triples, mini_embeddings)
        assert data.x.shape[0] == 3  # alice, bob, carol

    def test_correct_edge_count(self, mini_triples, mini_embeddings) -> None:
        """edge_index should have one column per triple."""
        from graphbench.gnn.subgraph import subgraph_to_pyg

        data = subgraph_to_pyg(mini_triples, mini_embeddings)
        assert data.edge_index.shape[1] == 2

    def test_node_index_attached(self, mini_triples, mini_embeddings) -> None:
        """Data object should have node_index and entities attributes."""
        from graphbench.gnn.subgraph import subgraph_to_pyg

        data = subgraph_to_pyg(mini_triples, mini_embeddings)
        assert hasattr(data, "node_index")
        assert hasattr(data, "entities")
        assert "alice" in data.node_index

    def test_zero_fallback(self, mini_triples) -> None:
        """Entities missing from embeddings should receive zero-vector features."""
        from graphbench.gnn.subgraph import subgraph_to_pyg

        data = subgraph_to_pyg(mini_triples, {})  # no embeddings
        import torch

        assert torch.all(data.x == 0)

    def test_raises_on_empty_triples(self, mini_embeddings) -> None:
        """Should raise ValueError for empty triple list."""
        from graphbench.gnn.subgraph import subgraph_to_pyg

        with pytest.raises(ValueError, match="must not be empty"):
            subgraph_to_pyg([], mini_embeddings)
