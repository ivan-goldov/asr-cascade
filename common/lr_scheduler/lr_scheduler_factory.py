from torch.optim.lr_scheduler import CosineAnnealingLR


def create_scheduler(optimizer, params):
    scheduler_kind = params.get_param("scheduler_kind")

    if scheduler_kind == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer=optimizer,
                                      T_max=params.get_param("num_iterations"),
                                      eta_min=params.get_param("min_lr"))
    elif scheduler_kind == "none":
        scheduler = None
    else:
        raise ValueError(f"Invalid scheduler choice: {scheduler_kind}")

    return scheduler
