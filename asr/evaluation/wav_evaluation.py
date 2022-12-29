import json
import logging
import os
import sys
from typing import Optional

import torch

from asr.evaluation import inference_model_from_checkpoint, GreedyGenerator, BeamGenerator
from asr.models import ModelResult, EncoderResult, DecoderResult
from asr.features_extractor import KaldiFeaturesExtractor
from common.decoder import create_decoder

LOG = logging.getLogger()


def identity(x):
    return x


def main(records_dir: str, model_path: str, nn_decoder: str, text_decoder: str, generator_type: str,
         beam_size: Optional[int], recognitions_path: str):
    LOG.info("loading model")

    model, features_config, dictionary = inference_model_from_checkpoint(model_path)

    generator = None
    if model._model_context["transformer_decoder"]:
        if generator_type == "greedy":
            generator = GreedyGenerator(model, dictionary)
        elif generator_type == "beam_search":
            generator = BeamGenerator(model, dictionary, beam_size)
        else:
            raise Exception(f"Unknown generator: {generator_type}")

    decoder = create_decoder(nn_decoder=nn_decoder,
                             text_decoder=text_decoder,
                             dictionary=dictionary)

    features_extractor = KaldiFeaturesExtractor(features_config, identity, identity)

    model.cuda()
    model.eval()

    LOG.info("recognition start")
    recognitions = recognize(records_dir, model, features_extractor, generator, decoder)

    LOG.info("saving result")
    with open(recognitions_path, "w") as f:
        json.dump(recognitions, f, indent=2, ensure_ascii=False)


def recognize(records_dir, model, features_extractor, generator, decoder, device="cuda"):
    recognitions = {}
    for i, record_id in enumerate(os.listdir(records_dir)):
        sys.stderr.write(f"#{i} recognizing record: {record_id}\n")

        with open(f"{records_dir}/{record_id}", "rb") as f:
            sample = features_extractor.extract_raw(f.read())

        sample = torch.tensor(sample).unsqueeze(0).to(device)
        sample = sample.transpose(1, 2).contiguous()

        with torch.no_grad():
            if model._model_context["transformer_decoder"]:
                prediction = generator(sample)
                transcript = decoder.dictionary().decode(prediction)
            else:
                encoder_result: EncoderResult = model.encoder(sample, None)
                decoder_result: DecoderResult = model.decoder(encoder_result, None, None)
                result = ModelResult(decoder_result.output, None, None)
                transcript = decoder.decode(result.log_probs)[0]

        recognitions[record_id] = transcript
        sys.stderr.write(f"{transcript}\n\n")

    return recognitions
