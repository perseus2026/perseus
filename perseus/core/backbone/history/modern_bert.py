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

        head_dim = dim // num_heads
        inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_events_per_sequence + 1, dtype=torch.float32).unsqueeze(dim=-1)
        freqs = inv_freq.unsqueeze(dim=0) * positions
        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

        self.layers = nn.ModuleList(EncoderLayer(dim, num_heads, dropout) for _ in range(num_layers))
        self.norm = nn.RMSNorm(dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        mask: torch.Tensor,
        /,
    ) -> torch.Tensor:
        x = embeddings
        cos, sin = self.cos[positions].unsqueeze(dim=1), self.sin[positions].unsqueeze(dim=1)
        mask = mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        self.attention = SelfAttention(dim, num_heads, dropout)
        self.ffn = SwiGLUFFN(dim, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x, cos, sin, mask)
        return x + self.ffn(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        x = self.norm(x)

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self._rotate(q, cos, sin)
        k = self._rotate(k, cos, sin)

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
        return self.dropout(x)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()

        hidden_dim = dim * 8 / 3
        hidden_dim = round(hidden_dim / 2) * 2

        self.norm = nn.RMSNorm(dim)
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = nn.functional.silu(gate) * up
        x = self.out_proj(x)
        return self.dropout(x)
