import typing as t

import torch
from torch import nn

from perseus import utils
from perseus.core import encoders
from perseus.core.backbone import context, event, history

__all__ = [
    "Backbone",
    "context",
    "event",
    "history",
]


class HistoryAggregatorConfig(t.TypedDict):
    max_events_per_sequence: int
    type: str
    params: dict[str, t.Any]


class EventAggregatorConfig(t.TypedDict):
    num_features: int
    type: str
    params: dict[str, t.Any]


class ContextAggregatorConfig(t.TypedDict):
    num_features: int
    type: str
    params: dict[str, t.Any]


class Backbone(utils.torch.Module):
    def __init__(
        self,
        dim: int,
        /,
        *,
        history_aggregator_config: HistoryAggregatorConfig,
        event_aggregator_config: EventAggregatorConfig,
        context_aggregator_config: ContextAggregatorConfig | None = None,
    ) -> None:
        super().__init__()

        self.history_aggregator = history.registry[history_aggregator_config["type"]](
            dim,
            history_aggregator_config["max_events_per_sequence"],
            **history_aggregator_config["params"],
        )
        self.event_aggregator = event.registry[event_aggregator_config["type"]](
            dim,
            event_aggregator_config["num_features"],
            **event_aggregator_config["params"],
        )
        if context_aggregator_config:
            self.context_aggregator = context.registry[context_aggregator_config["type"]](
                dim,
                context_aggregator_config["num_features"],
                **context_aggregator_config["params"],
            )

        self.readout_token = nn.Parameter(torch.empty(dim))
        nn.init.trunc_normal_(self.readout_token, std=utils.torch.INIT_WEIGHTS_STD)

    def forward(
        self,
        feature_to_embedder: dict[str, encoders.Embedder],
        events_features: dict[str, torch.Tensor],
        events_positions: torch.Tensor,
        events_timestamps: torch.Tensor,
        sample_timestamp: torch.Tensor,
        context_features: dict[str, torch.Tensor] | None = None,
        /,
    ) -> torch.Tensor:
        embeddings, num_context_tokens = self._build_embeddings(feature_to_embedder, events_features, context_features)
        positions = self._build_positions(events_positions, num_context_tokens)
        timestamps = self._build_timestamps(events_timestamps, sample_timestamp, num_context_tokens)
        mask = self._build_mask(events_positions, num_context_tokens)
        return self.history_aggregator(embeddings, positions, timestamps, mask)[:, -1]

    def _build_embeddings(
        self,
        feature_to_embedder: dict[str, encoders.Embedder],
        events_features: dict[str, torch.Tensor],
        context_features: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, int]:
        events_embeddings = torch.stack(
            [feature_to_embedder[feature](values) for feature, values in events_features.items()],
            dim=2,
        )
        embeddings = self.event_aggregator(events_embeddings)
        num_context_tokens = 0
        if context_features:
            context_embeddings = torch.stack(
                [feature_to_embedder[feature](values) for feature, values in context_features.items()],
                dim=1,
            )
            aggregated_context = self.context_aggregator(context_embeddings)
            num_context_tokens = aggregated_context.size(1)
            embeddings = torch.cat([embeddings, aggregated_context], dim=1)

        embeddings = torch.cat(
            [embeddings, self.readout_token.expand(embeddings.size(0), 1, embeddings.size(2))],
            dim=1,
        )
        return embeddings, num_context_tokens

    def _build_positions(self, events_positions: torch.Tensor, num_context_tokens: int) -> torch.Tensor:
        context_readout_positions = torch.zeros_like(events_positions[:, :1])
        if num_context_tokens > 0:
            context_readout_positions = context_readout_positions.expand(-1, 1 + num_context_tokens)
        return torch.cat([events_positions, context_readout_positions], dim=1)

    def _build_timestamps(
        self,
        events_timestamps: torch.Tensor,
        sample_timestamp: torch.Tensor,
        num_context_tokens: int,
    ) -> torch.Tensor:
        context_readout_timestamps = sample_timestamp.unsqueeze(dim=1)
        if num_context_tokens > 0:
            context_readout_timestamps = context_readout_timestamps.expand(-1, 1 + num_context_tokens)
        return torch.cat([events_timestamps, context_readout_timestamps], dim=-1)

    def _build_mask(self, events_positions: torch.Tensor, num_context_tokens: int) -> torch.Tensor:
        events_padding_mask = events_positions == 0
        context_readout_padding_mask = torch.zeros_like(events_padding_mask[:, :1])
        if num_context_tokens > 0:
            context_readout_padding_mask = context_readout_padding_mask.expand(-1, 1 + num_context_tokens)
        padding_mask = torch.cat([events_padding_mask, context_readout_padding_mask], dim=-1)

        total_len = events_positions.size(1) + num_context_tokens + 1
        attention_mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=events_positions.device)
        attention_mask[:-1, -1] = True

        mask = padding_mask.unsqueeze(1) | attention_mask.unsqueeze(0)
        diag = torch.arange(mask.size(-1), device=mask.device)
        mask[:, diag, diag] = False
        return mask
