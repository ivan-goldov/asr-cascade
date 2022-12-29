import torch

from asr.models import EncoderDecoderModel
from asr.progress_tracker import ASRProgressTracker

from lm.models import GPT


def inference_model_from_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    progress = ASRProgressTracker()
    progress.load(checkpoint["progress_tracker_state"])
    last_training_params = progress.last_training_params()
    features_config = last_training_params["features_config"]
    model_definition = last_training_params["model_definition"]

    for layer_config in model_definition["encoder"]:
        if layer_config["name"] == "jasper_encoder":
            layer_config["convmask"] = False

    dictionary = last_training_params["dictionary"]

    language_model = GPT(model_definition["language_model"], dictionary)
    model = EncoderDecoderModel(model_definition, features_config["num-mel-bins"], dictionary, language_model)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return model, features_config, dictionary
