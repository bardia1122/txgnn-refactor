"""R-GCN encoder + DistMult decoder — the lightweight structure model.

Matches what the DrKGC paper used for PrimeKG ("we used R-GCN with our optimal
hyperparameter settings to obtain global embeddings", appendix A.5): a
relational GCN over the training KG, scored with DistMult.

The encoder output is what later steps consume as `E_global`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    """Learnable entity embeddings refined by `num_layers` relational GCN layers."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 200,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs = nn.ModuleList(
            RGCNConv(dim, dim, num_relations, num_bases=num_bases)
            for _ in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DistMultDecoder(nn.Module):
    """score(h, r, t) = sum(h * r_embedding * t)."""

    def __init__(self, num_relations: int, dim: int) -> None:
        super().__init__()
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation.weight)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (h * self.relation(r) * t).sum(dim=-1)

    def score_against(
        self, anchor: torch.Tensor, r: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Score one anchor per row against a shared candidate pool.

        anchor: [B, D] (the known entity), r: [B], candidates: [C, D]
        returns [B, C].
        """
        return (anchor * self.relation(r)) @ candidates.t()


class RGCNRanker(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 200,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # inverse relations exist only for message passing; scoring uses the
        # forward half, but sizing the decoder to match keeps the ids aligned
        self.encoder = RGCNEncoder(
            num_entities, num_relations, dim, num_layers, num_bases, dropout
        )
        self.decoder = DistMultDecoder(num_relations, dim)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        return self.encoder(edge_index, edge_type)

    def score(self, z: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        return self.decoder(z[h], r, z[t])
