import multiprocessing

import torch
from cyclopts import App
from dotenv import load_dotenv

from perseus import utils
from perseus.cli import event_hub, inference, train

__all__ = [
    "run_app",
]


def run_app() -> None:
    multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_start_method("spawn", force=True)
    load_dotenv(override=True)
    utils.logging.configure()

    app = App()
    app.command(event_hub.app, name="event-hub")
    app.command(train.app, name="train")
    app.command(inference.app, name="inference")
    app()
