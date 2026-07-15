import typing as t

import transformers
from torch import optim


class SchedulerFactory(t.Protocol):
    def __call__(self, optimizer: optim.Optimizer, *args: t.Any, **kwargs: t.Any) -> optim.lr_scheduler.LRScheduler: ...


registry: dict[str, SchedulerFactory] = {
    "constant": transformers.get_constant_schedule,
    "constant_with_warmup": transformers.get_constant_schedule_with_warmup,
    "linear": transformers.get_linear_schedule_with_warmup,
    "cosine": transformers.get_cosine_schedule_with_warmup,
    "cosine_with_restarts": transformers.get_cosine_with_hard_restarts_schedule_with_warmup,
    "cosine_with_min_lr": transformers.get_cosine_with_min_lr_schedule_with_warmup,
    "cosine_warmup_with_min_lr": transformers.get_cosine_with_min_lr_schedule_with_warmup_lr_rate,
    "polynomial": transformers.get_polynomial_decay_schedule_with_warmup,
    "inverse_sqrt": transformers.get_inverse_sqrt_schedule,
    "warmup_stable_decay": transformers.get_wsd_schedule,
}
