import torch
from torch import nn

from perseus.core.backbone.history import base


class Aggregator(base.Aggregator):
    def __init__(
        self,
        dim: int,
        max_events_per_sequence: int,
        /,
        *,
        num_heads: int = 1,
        dropout: float = 0.1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(dim, max_events_per_sequence)

        self.num_heads = num_heads
        self.positions_embeddings = nn.Embedding(max_events_per_sequence + 1, dim, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            norm=nn.LayerNorm(dim),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        mask: torch.Tensor,
        /,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.shape
        mask = (
            mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, seq_len, seq_len)
        )
        return self.encoder(embeddings + self.positions_embeddings(positions), mask)
