from abc import ABC, abstractmethod

import torch
from torch import nn


class Loss(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: torch.Tensor,
        logq: torch.Tensor,
    ) -> torch.Tensor: ...


class CrossEntropy(Loss):
    def __init__(self, *, logq_correction: bool = False) -> None:
        super().__init__()

        self.logq_correction = logq_correction

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: torch.Tensor,
        logq: torch.Tensor,
    ) -> torch.Tensor:
        logits = user_embeddings @ item_embeddings.T
        if self.logq_correction:
            logits = logits - logq
        return nn.functional.cross_entropy(logits, labels)


class ScalableCrossEntropy(Loss):
    def __init__(
        self,
        *,
        num_buckets: int,
        x_bucket_size: int,
        y_bucket_size: int,
        mix: bool = True,
    ) -> None:
        super().__init__()

        self.num_buckets = num_buckets
        self.x_bucket_size = x_bucket_size
        self.y_bucket_size = y_bucket_size
        self.mix = mix

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: torch.Tensor,
        _logq: torch.Tensor,
    ) -> torch.Tensor:
        bucket_centers = self._build_bucket_centers(user_embeddings)
        bucket_user_indices, bucket_item_indices = self._assign_to_buckets(
            bucket_centers,
            user_embeddings,
            item_embeddings,
            labels,
        )
        return self._calculate_loss(
            bucket_user_indices,
            bucket_item_indices,
            user_embeddings,
            item_embeddings,
            labels,
        )

    def _build_bucket_centers(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        if self.mix:
            omega = torch.randn(
                user_embeddings.size(0),
                self.num_buckets,
                dtype=torch.float32,
                device=user_embeddings.device,
            )
            with torch.no_grad():
                buckets = omega.T @ user_embeddings
        else:
            buckets = torch.randn(
                self.num_buckets,
                user_embeddings.size(1),
                dtype=torch.float32,
                device=user_embeddings.device,
            )
        buckets /= user_embeddings.size(1) ** 0.25
        return buckets

    def _assign_to_buckets(
        self,
        bucket_centers: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            bucket_user_dists: torch.Tensor = bucket_centers @ user_embeddings.T
            bucket_item_dists: torch.Tensor = bucket_centers @ item_embeddings.T

        _, bucket_user_indices = bucket_user_dists.topk(min(self.x_bucket_size, bucket_user_dists.size(1)))

        item_indices = torch.arange(len(item_embeddings), dtype=torch.int32, device=item_embeddings.device)
        positive_item_indices = item_indices[(labels[bucket_user_indices.unique()] > 0.0).any(dim=0)]
        negative_item_mask = torch.ones(len(item_embeddings), dtype=torch.bool, device=item_embeddings.device)
        negative_item_mask[positive_item_indices] = False
        bucket_item_dists = bucket_item_dists[:, negative_item_mask]
        _, bucket_negative_item_indices = bucket_item_dists.topk(min(self.y_bucket_size, bucket_item_dists.size(1)))
        bucket_item_indices = torch.cat(
            [
                bucket_negative_item_indices,
                positive_item_indices.unsqueeze(dim=0).expand(bucket_negative_item_indices.size(0), -1),
            ],
            dim=-1,
        )

        return bucket_user_indices, bucket_item_indices

    def _calculate_loss(
        self,
        bucket_user_indices: torch.Tensor,
        bucket_item_indices: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        bucket_user_embeddings = user_embeddings[bucket_user_indices]
        bucket_item_embeddings = item_embeddings[bucket_item_indices]
        bucket_logits = bucket_user_embeddings @ bucket_item_embeddings.transpose(-1, -2)

        bucket_labels = labels[bucket_user_indices.unsqueeze(dim=2), bucket_item_indices.unsqueeze(dim=1)]

        bucket_user_loss = nn.functional.cross_entropy(
            bucket_logits.reshape(-1, bucket_logits.size(-1)),
            bucket_labels.reshape(-1, bucket_labels.size(-1)),
            reduction="none",
        )
        user_loss = torch.zeros(
            user_embeddings.size(0),
            device=bucket_user_loss.device,
            dtype=bucket_user_loss.dtype,
        )
        user_loss.scatter_reduce_(
            dim=0,
            index=bucket_user_indices.reshape(-1),
            src=bucket_user_loss.reshape(-1),
            reduce="amax",
        )
        user_loss = user_loss[user_loss != 0.0]
        return user_loss.mean()
