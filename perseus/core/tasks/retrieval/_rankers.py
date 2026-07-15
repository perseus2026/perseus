from abc import ABC, abstractmethod

import torch


class Ranker(ABC):
    def __init__(self, item_embeddings: torch.Tensor, /) -> None:
        self.item_embeddings = item_embeddings

    @abstractmethod
    def top(self, logits: torch.Tensor, /, *, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Naive(Ranker):
    def top(self, logits: torch.Tensor, /, *, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = logits.topk(k)
        return top_k.indices, top_k.values


class Smmr(Ranker):
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        /,
        *,
        pool_size: int | None = None,
        lambda_: float = 0.5,
        scale_factor: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(item_embeddings)

        self.pool_size = pool_size
        self.lambda_ = lambda_
        self.scale_factor = scale_factor
        self.temperature = temperature

    def top(self, logits: torch.Tensor, /, *, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        pool_size = k if self.pool_size is None else self.pool_size
        if k > pool_size:
            raise ValueError(f"k must be equal or lower than pool size, but {k} > {pool_size}")
        pool_size = min(pool_size, logits.size(-1))
        pool_logits, pool_indices = logits.topk(pool_size)

        pool_probas = pool_logits.sigmoid()
        pool_item_embeddings = self.item_embeddings[pool_indices]
        pool_item_embeddings /= pool_item_embeddings.norm(p=2, dim=-1).unsqueeze(dim=-1)
        pool_similarities = pool_item_embeddings @ pool_item_embeddings.permute(0, 2, 1)

        selected_indices = pool_probas.argmax(dim=1).unsqueeze(dim=-1)
        row_index = torch.arange(
            len(selected_indices),
            dtype=torch.int32,
            device=selected_indices.device,
        ).unsqueeze(dim=-1)
        pool_item_is_selected = torch.full(
            size=(len(logits), pool_size),
            fill_value=False,
            dtype=torch.bool,
            device=logits.device,
        )
        while (num_selected := selected_indices.size(1)) < k:
            pool_item_is_selected[row_index, selected_indices] = True
            batch_max_similarity_scores = (
                pool_similarities[pool_item_is_selected]
                .view(
                    pool_item_is_selected.size(0),
                    -1,
                    pool_item_is_selected.size(1),
                )
                .max(dim=1)
                .values
            )

            batch_mmr_scores = self.lambda_ * pool_probas - (1 - self.lambda_) * batch_max_similarity_scores
            batch_mmr_scores[pool_item_is_selected] = -torch.inf
            batch_mmr_probas = _softmax(batch_mmr_scores, temperature=self.temperature)
            batch_mmr_probas += ~pool_item_is_selected * 1e-30

            batch_size = max(1, min(int(num_selected * (self.scale_factor - 1)), (k - num_selected)))
            batch_selected_indices = torch.multinomial(batch_mmr_probas, batch_size, replacement=False)
            selected_indices = torch.cat((selected_indices, batch_selected_indices), dim=-1)

        return pool_indices[row_index, selected_indices], pool_logits[row_index, selected_indices]


def _softmax(x: torch.Tensor, /, *, temperature: float = 1.0) -> torch.Tensor:
    x = x - x.max(dim=-1).values.unsqueeze(dim=-1)  # for numerical stability
    x = (x / temperature).exp()
    return x / x.sum(dim=-1).unsqueeze(dim=-1)
