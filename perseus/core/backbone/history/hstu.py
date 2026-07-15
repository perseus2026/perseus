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
        num_buckets: int = 128,
        num_heads: int = 1,
        dropout: float = 0.1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(dim, max_events_per_sequence)

        self.num_positions = max_events_per_sequence
        self.num_buckets = num_buckets
        self.position_bias = nn.Embedding(self.num_positions * 2 + 1, 1, padding_idx=0)
        self.timestamp_bias = nn.Embedding(self.num_buckets * 2 + 1, 1, padding_idx=0)

        self.layers = nn.ModuleList(EncoderLayer(dim, num_heads, dropout) for _ in range(num_layers))

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
        /,
    ) -> torch.Tensor:
        x = embeddings
        rab = self._prepare_rab(positions, timestamps, mask)
        mask = mask.unsqueeze(1)
        scale = self._prepare_scale(timestamps)
        for layer in self.layers:
            x = layer(x, mask, rab, scale)
        return x

    def _prepare_rab(self, positions: torch.Tensor, timestamps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions_diff = positions.unsqueeze(dim=2) - positions.unsqueeze(dim=1)
        timestamps_diff = timestamps.unsqueeze(dim=2) - timestamps.unsqueeze(dim=1)

        positions_bucket = torch.zeros_like(positions_diff, dtype=torch.int32)
        positions_bucket[positions_diff >= 0] = positions_diff[positions_diff >= 0] + 1
        positions_bucket[positions_diff < 0] = positions_diff[positions_diff < 0].neg() + self.num_positions
        positions_bucket *= ~mask

        timestamps_bucket = torch.zeros_like(timestamps_diff, dtype=torch.int32)
        timestamps_bucket[timestamps_diff >= 0] = (timestamps_diff[timestamps_diff >= 0].log1p() / 0.301).int()
        timestamps_bucket[timestamps_diff < 0] = (
            timestamps_diff[timestamps_diff < 0].neg().log1p() / 0.301
        ).int() + self.num_buckets
        timestamps_bucket *= ~mask

        positions_bias = self.position_bias(positions_bucket)
        timestamps_bias = self.timestamp_bias(timestamps_bucket)
        return (positions_bias + timestamps_bias).squeeze(dim=-1).unsqueeze(dim=1)

    def _prepare_scale(self, timestamps: torch.Tensor) -> torch.Tensor:
        return (timestamps != 0.0).sum(dim=-1).view(timestamps.size(0), 1, 1, 1)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_norm = nn.LayerNorm(dim)
        self.uvqk_proj = nn.Linear(dim, 4 * dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, rab: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        u, v, q, k = self._point_wise_projection(x)
        av = self._spatial_aggregation(v, q, k, mask, rab, scale)
        y = self._pointwise_transformation(av, u)
        return x + y

    def _point_wise_projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        u, v, q, k = nn.functional.silu(self.uvqk_proj(self.input_norm(x))).chunk(4, dim=-1)
        u = u.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return u, v, q, k

    def _spatial_aggregation(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        mask: torch.Tensor,
        rab: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        scores = q @ k.transpose(-2, -1) + rab
        weights = (nn.functional.silu(scores) / scale).masked_fill(mask, 0.0)
        return weights @ v

    def _pointwise_transformation(self, av: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = av.shape
        av = av.transpose(1, 2).reshape(batch_size, seq_len, -1)
        u = u.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.output_proj(self.dropout(self.output_norm(av) * u))
