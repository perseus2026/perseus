from functools import partial

import torch
from torch import nn

try:
    from mamba_ssm import Mamba2
    from mamba_ssm.modules.block import Block
    from mamba_ssm.modules.mlp import GatedMLP
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn
except ImportError:
    Mamba2 = Block = GatedMLP = RMSNorm = rms_norm_fn = None

from perseus.core.backbone.history import base


class Aggregator(base.Aggregator):
    def __init__(
        self,
        dim: int,
        max_events_per_sequence: int,
        /,
        *,
        head_dim: int = 64,
        state_dim: int = 64,
        expand: int = 2,
        conv_kernel: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__(dim, max_events_per_sequence)

        self.blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            mixer_cls = partial(
                Mamba2,
                layer_idx=layer_idx,
                d_state=state_dim,
                d_conv=conv_kernel,
                expand=expand,
                headdim=head_dim,
            )
            mlp_cls = partial(
                GatedMLP,
                hidden_features=round(dim * 8 / 3 / 2) * 2,
                out_features=dim,
                multiple_of=2,
            )
            block = Block(
                dim,
                mixer_cls,
                mlp_cls,
                norm_cls=RMSNorm,
                fused_add_norm=True,
                residual_in_fp32=True,
            )
            block.layer_idx = layer_idx
            self.blocks.append(block)
        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        _positions: torch.Tensor,
        _timestamps: torch.Tensor,
        _mask: torch.Tensor,
        /,
    ) -> torch.Tensor:
        x, residual = embeddings, None
        for layer in self.blocks:
            x, residual = layer(x, residual=residual)
        return rms_norm_fn(
            x,
            self.final_norm.weight,
            self.final_norm.bias,
            eps=self.final_norm.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
