from .cross_entropy import CrossEntropyLoss
from .ctc import CTCLossNM


def create_objective(dictionary, params):
    objective_type = params.get_param("objective")
    if objective_type == "ctc_loss":
        objective = CTCLossNM(blank_id=dictionary.blank_id())
    elif objective_type == "cross_entropy_loss":
        objective = CrossEntropyLoss(pad_id=dictionary.pad_id())
    else:
        raise Exception(f"Unknown objective function: {objective_type}")

    return objective
