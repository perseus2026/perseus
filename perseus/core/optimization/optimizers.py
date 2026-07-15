from torch import optim

registry: dict[str, type[optim.Optimizer]] = {
    "adadelta": optim.Adadelta,
    "adafactor": optim.Adafactor,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "sgd": optim.SGD,
}
