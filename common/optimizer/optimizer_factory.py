from .adam import AdamW
from .novograd import Novograd


def create_optimizer(model, params):
    optimizer_kind = params["optimizer_kind"]

    if optimizer_kind == "novograd":
        optimizer = Novograd(model.parameters(),
                             lr=params["lr"],
                             weight_decay=params["weight_decay"])
    elif optimizer_kind == "adam":
        optimizer = AdamW(model.parameters(),
                          lr=params["lr"],
                          weight_decay=params["weight_decay"])
    else:
        raise ValueError(f"Invalid optimizer choice: {optimizer_kind}")

    return optimizer
