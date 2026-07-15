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

        self.position_embeddings = nn.Embedding(max_events_per_sequence + 1, dim, padding_idx=0)
        self.layers = nn.ModuleList(EncoderLayer(dim, num_heads, dropout) for _ in range(num_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        mask: torch.Tensor,
        /,
    ) -> torch.Tensor:
        x = embeddings + self.position_embeddings(positions)
        mask = mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        self.attention = SelfAttention(dim, num_heads, dropout)
        self.ffn = SwiGLUFFN(dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x, mask)
        return x + self.ffn(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        gate = nn.functional.sigmoid(self.gate(x))

        x = self.norm(x)

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        x = (
            nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=~mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )
            .transpose(1, 2)
            .reshape(batch_size, seq_len, dim)
        )

        x = self.out_proj(x)
        return gate * self.dropout(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()

        hidden_dim = dim * 8 / 3
        hidden_dim = round(hidden_dim / 2) * 2

        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.sigmoid(self.gate(x))

        x = self.norm(x)
        up_gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = nn.functional.silu(up_gate) * up
        x = self.out_proj(x)
        return gate * self.dropout(x)
