# model.py

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.dropout = dropout

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Encode nodes into embeddings.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return x


class InnerProductDecoder(nn.Module):
    """
    Inner-product edge decoder.

    For each edge (u, v), score = <z_u, z_v>.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor, edge_pairs: Tensor) -> Tensor:
        """
        Args:
            z         : [num_nodes, hidden_channels]
            edge_pairs: [num_edges, 2]

        Returns:
            logits    : [num_edges]
        """
        src = edge_pairs[:, 0]
        dst = edge_pairs[:, 1]

        z_src = z[src]
        z_dst = z[dst]

        logits = torch.sum(z_src * z_dst, dim=-1)
        return logits


class GraphSAGELinkPredictor(nn.Module):
    """
    Full GraphSAGE link prediction model:
    encoder + inner-product decoder
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.decoder = InnerProductDecoder()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Get node embeddings.
        """
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_pairs: Tensor) -> Tensor:
        """
        Get edge logits.
        """
        return self.decoder(z, edge_pairs)

    def decode_proba(self, z: Tensor, edge_pairs: Tensor) -> Tensor:
        """
        Get edge probabilities after sigmoid.
        """
        logits = self.decode(z, edge_pairs)
        probs = torch.sigmoid(logits)
        return probs

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_pairs: Tensor,
    ) -> Tensor:
        """
        End-to-end forward pass.

        Returns edge logits.
        """
        z = self.encode(x, edge_index)
        logits = self.decode(z, edge_pairs)
        return logits


def build_bce_loss(
    pos_logits: Tensor,
    neg_logits: Tensor,
) -> Tensor:
    """
    Binary cross-entropy loss for positive and negative edges.
    """
    logits = torch.cat([pos_logits, neg_logits], dim=0)

    labels = torch.cat(
        [
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ],
        dim=0,
    )

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss


@torch.no_grad()
def compute_edge_scores(
    model: GraphSAGELinkPredictor,
    x: Tensor,
    edge_index: Tensor,
    edge_pairs: Tensor,
) -> Tensor:
    """
    Compute edge probabilities in eval mode.
    """
    model.eval()
    z = model.encode(x, edge_index)
    probs = model.decode_proba(z, edge_pairs)
    return probs


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    """
    Quick smoke test with fake tensors.
    """
    num_nodes = 10
    in_channels = 4
    hidden_channels = 32

    x = torch.randn(num_nodes, in_channels)

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )

    edge_pairs = torch.tensor(
        [
            [0, 2],
            [1, 3],
            [4, 5],
        ],
        dtype=torch.long,
    )

    model = GraphSAGELinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dropout=0.3,
    )

    logits = model(x, edge_index, edge_pairs)
    probs = torch.sigmoid(logits)

    print("logits shape:", logits.shape)
    print("probs shape :", probs.shape)
    print("num params  :", count_parameters(model))


if __name__ == "__main__":
    main()