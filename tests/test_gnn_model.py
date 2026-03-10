"""Tests for graphbench.gnn.model.

Full implementation in Phase 3.
"""

import pytest


@pytest.mark.skip(reason="Phase 3 — not yet implemented")
def test_gat_model_forward_pass() -> None:
    """GAT model forward pass should return edge scores of correct shape."""
    pass


@pytest.mark.skip(reason="Phase 3 — not yet implemented")
def test_gat_model_output_in_0_1() -> None:
    """GAT model output (after sigmoid) should be in [0, 1]."""
    pass


@pytest.mark.skip(reason="Phase 3 — not yet implemented")
def test_gat_model_layer_count() -> None:
    """GAT model should have exactly 3 GATConv layers."""
    pass


@pytest.mark.skip(reason="Phase 3 — not yet implemented")
def test_gat_model_attention_heads() -> None:
    """Each GAT layer should have 4 attention heads."""
    pass
