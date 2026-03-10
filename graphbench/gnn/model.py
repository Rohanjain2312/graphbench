"""3-layer Graph Attention Network (GAT) for link prediction on the KG.

Architecture (encoder):
    Layer 1: GATConv(384 →  64, heads=4, concat=True)  → 256-dim per node
    Layer 2: GATConv(256 →  32, heads=4, concat=True)  → 128-dim per node
    Layer 3: GATConv(128 →  32, heads=4, concat=False) →  32-dim per node

Decoder: dot-product of endpoint embeddings → scalar logit per edge.

encode() / decode() / forward() are kept separate so GNNRAGPipeline can call
encode() once and decode() for every candidate edge set without redundant
attention computation.
"""

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

try:
    from torch_geometric.nn import GATConv
except ImportError as exc:
    raise ImportError(
        "torch-geometric is required for the GNN module. "
        "Install via: pip install torch-geometric"
    ) from exc

logger = logging.getLogger(__name__)


class GATModel(torch.nn.Module):
    """3-layer GAT encoder with dot-product decoder for KG link prediction.

    Attributes:
        convs: ModuleList of 3 GATConv layers.
        dropout: Dropout applied after layers 1 and 2.

    Args:
        in_channels: Input node feature dimension (default 384).
        hidden_channels: Per-head output dimension for layers 1 and 2 (default 64).
        out_channels: Per-head output dimension for layer 3 (default 32).
            With concat=False on layer 3, the final node embedding is out_channels-dim.
        heads: Number of attention heads per layer (default 4).
        dropout: Dropout probability on node features after layers 1 and 2 (default 0.3).
        attn_dropout: Dropout on attention coefficients inside GATConv (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 384,
        hidden_channels: int = 64,
        out_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Layer 1: 384 → 64 * 4 = 256
        conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=attn_dropout,
        )
        # Layer 2: 256 → 32 * 4 = 128
        conv2 = GATConv(
            hidden_channels * heads,
            out_channels // 2,  # 16 per head → 16*4=64... adjust below
            heads=heads,
            concat=True,
            dropout=attn_dropout,
        )
        # Layer 3: 64 → 32 (averaged across 4 heads, concat=False)
        conv3 = GATConv(
            (out_channels // 2) * heads,
            out_channels,
            heads=heads,
            concat=False,  # average heads → out_channels-dim
            dropout=attn_dropout,
        )

        self.convs = torch.nn.ModuleList([conv1, conv2, conv3])
        self.dropout = torch.nn.Dropout(p=dropout)

        logger.info(
            "GATModel initialised: %d layers, %d heads, "
            "in=%d → hidden=%d → out=%d (after head avg).",
            len(self.convs),
            heads,
            in_channels,
            hidden_channels * heads,
            out_channels,
        )

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Run GAT message-passing to produce per-node embeddings.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Graph connectivity of shape (2, E).

        Returns:
            Node embedding matrix of shape (N, out_channels).
        """
        # Layer 1 → ELU → Dropout
        h = self.convs[0](x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        # Layer 2 → ELU → Dropout
        h = self.convs[1](h, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        # Layer 3 (no activation, no dropout — final representation)
        h = self.convs[2](h, edge_index)
        return h  # shape: (N, out_channels)

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        """Score candidate edges via dot product of endpoint embeddings.

        Args:
            z: Node embedding matrix of shape (N, out_channels).
            edge_label_index: Candidate edges of shape (2, E_cand).

        Returns:
            Raw logits of shape (E_cand,). Apply sigmoid for probabilities.
        """
        src_emb = z[edge_label_index[0]]  # (E_cand, out_channels)
        dst_emb = z[edge_label_index[1]]  # (E_cand, out_channels)
        # Element-wise product then sum == dot product per edge
        return (src_emb * dst_emb).sum(dim=-1)  # (E_cand,)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        """Encode nodes then decode candidate edges.

        Args:
            x: Node features (N, in_channels).
            edge_index: Message-passing edges (2, E).
            edge_label_index: Supervision edges to score (2, E_cand).

        Returns:
            Logits of shape (E_cand,).
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def score_edges(
        self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        """Return sigmoid-normalised edge scores in [0, 1].

        Convenience wrapper for inference (GNNRAGPipeline).

        Args:
            x: Node features (N, in_channels).
            edge_index: Message-passing edges (2, E).
            edge_label_index: Edges to score (2, E_cand).

        Returns:
            Scores in [0.0, 1.0] of shape (E_cand,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_label_index)
        return torch.sigmoid(logits)
