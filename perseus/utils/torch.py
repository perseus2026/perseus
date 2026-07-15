import inspect
import json
import typing as t
from functools import wraps
from pathlib import Path

import torch
from accelerate import Accelerator, PartialState
from safetensors.torch import load_file, save_file
from torch import nn

INIT_WEIGHTS_STD = 0.02

device: torch.device = PartialState(cpu=not torch.cuda.is_available()).device


class Module(nn.Module):
    _config: dict[str, t.Any]
    _wrapped: t.Self

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)

        original_init = cls.__init__

        @wraps(original_init)
        def wrapped_init(self: t.Self, *args: t.Any, **kwargs: t.Any) -> None:
            if "_config" not in self.__dict__:
                signature = inspect.signature(original_init)
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                pos: list[t.Any] = []
                kw: dict[str, t.Any] = {}
                for name, value in bound.arguments.items():
                    if name == "self":
                        continue
                    match signature.parameters[name].kind:
                        case inspect.Parameter.POSITIONAL_ONLY:
                            pos.append(value)
                        case inspect.Parameter.VAR_POSITIONAL:
                            pos.extend(value)
                        case inspect.Parameter.VAR_KEYWORD:
                            kw.update(value)
                        case _:
                            kw[name] = value
                self._config = {"args": pos, "kwargs": kw}

            original_init(self, *args, **kwargs)

            self.apply(init_weights)

        cls.__init__ = wrapped_init

    def wrap(self, accelerator: Accelerator, /) -> None:
        wrapped = self.__dict__.get("_wrapped")
        if wrapped is not None:
            raise RuntimeError("module already wrapped")
        object.__setattr__(self, "_wrapped", accelerator.prepare_model(self))

    def unwrap(self, accelerator: Accelerator, /) -> None:
        wrapped = self.__dict__.get("_wrapped")
        if wrapped is None:
            raise RuntimeError("module not wrapped")
        accelerator.unwrap_model(wrapped, keep_fp32_wrapper=False)
        self.__dict__.pop("_wrapped")

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        wrapped = self.__dict__.get("_wrapped")
        if wrapped is not None and not self.__dict__.get("_dispatching", False):
            object.__setattr__(self, "_dispatching", True)
            try:
                return wrapped(*args, **kwargs)
            finally:
                object.__setattr__(self, "_dispatching", False)
        return super().__call__(*args, **kwargs)

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(self._config, indent=4))
        save_file(self.state_dict(), path / "weights.safetensors")

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        config = json.loads((path / "config.json").read_text())
        self = cls(*config["args"], **config["kwargs"])
        state_dict = load_file(path / "weights.safetensors")
        self.load_state_dict(state_dict)
        return self


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=INIT_WEIGHTS_STD)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=INIT_WEIGHTS_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.RMSNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
    elif isinstance(module, nn.MultiheadAttention):
        for weight in (module.in_proj_weight, module.q_proj_weight, module.k_proj_weight, module.v_proj_weight):
            if weight is not None:
                nn.init.trunc_normal_(weight, std=INIT_WEIGHTS_STD)
        for bias in (module.in_proj_bias, module.bias_k, module.bias_v):
            if bias is not None:
                nn.init.zeros_(bias)
