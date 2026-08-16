"""The GCN adapter: subgraph -> structure-aware embeddings for the LLM.

DrKGC section 3.5. Two pieces:

**SubgraphGCN** — a low-dimensional relational GCN. It is initialised with the
*global* embeddings of the subgraph's entities and propagates over the retrieved
edges, so an entity's output depends on the query-specific local structure. The
paper runs this in a low dimension deliberately ("to reduce computational
overhead for graphs, GCN computations are performed in a low-dimensional space").

**StructureAdapter** — projects `[global ; local]` to the LLM's input width, so
the vectors can be spliced into the prompt in place of the `[Placeholder]` tokens.

Neither has a loss. They are trained jointly with the LLM in step 5, where the
paper allows gradients to flow through the whole model including this adapter.
That is why this module is deliberately small and side-effect free.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class SubgraphGCN(nn.Module):
    """Global embeddings -> query-specific local embeddings."""

    def __init__(
        self,
        global_dim: int = 200,
        hidden_dim: int = 128,
        num_relations: int = 12,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # project into the low-dimensional space the GCN operates in
        self.input_projection = nn.Linear(global_dim, hidden_dim)
        self.convs = nn.ModuleList(
            RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            for _ in range(num_layers)
        )
        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def forward(
        self,
        global_features: torch.Tensor,  # [N, global_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_type: torch.Tensor,  # [E]
    ) -> torch.Tensor:
        x = self.input_projection(global_features)
        for i, conv in enumerate(self.convs):
            # an isolated node (unreachable candidate) simply keeps its projected
            # features - RGCNConv's root weight handles the empty-neighbourhood case
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class StructureAdapter(nn.Module):
    """`[global ; local]` -> LLM input width."""

    def __init__(
        self,
        global_dim: int = 200,
        hidden_dim: int = 128,
        llm_dim: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(global_dim + hidden_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )
        self.layer_norm = nn.LayerNorm(llm_dim)
        self.llm_dim = llm_dim

    def forward(self, global_emb: torch.Tensor, local_emb: torch.Tensor) -> torch.Tensor:
        enhanced = torch.cat([global_emb, local_emb], dim=-1)
        # normalised so the injected vectors sit on the scale of the LLM's own
        # token embeddings; without this they can dominate early in fine-tuning
        return self.layer_norm(self.projection(enhanced))


class GCNAdapter(nn.Module):
    """The full step-4 component: subgraph in, prompt-ready vectors out."""

    def __init__(
        self,
        global_dim: int = 200,
        hidden_dim: int = 128,
        llm_dim: int = 4096,
        num_relations: int = 12,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gcn = SubgraphGCN(
            global_dim, hidden_dim, num_relations, num_layers, num_bases, dropout
        )
        self.adapter = StructureAdapter(global_dim, hidden_dim, llm_dim, dropout)

    def forward(
        self,
        global_features: torch.Tensor,  # [N, global_dim] for the batched subgraphs
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        query_index: torch.Tensor,  # [B]
        candidate_index: torch.Tensor,  # [B, k]
    ) -> dict:
        """Returns the vectors the prompt needs, one per placeholder.

        `query`      [B, llm_dim]
        `candidates` [B, k, llm_dim]
        """
        local = self.gcn(global_features, edge_index, edge_type)

        query_vectors = self.adapter(
            global_features[query_index], local[query_index]
        )
        flat = candidate_index.reshape(-1)
        candidate_vectors = self.adapter(global_features[flat], local[flat])
        candidate_vectors = candidate_vectors.view(
            *candidate_index.shape, self.adapter.llm_dim
        )
        return {
            "query": query_vectors,
            "candidates": candidate_vectors,
            "local": local,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
