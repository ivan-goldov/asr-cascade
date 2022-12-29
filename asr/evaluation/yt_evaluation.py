import json
import logging
import os
import sys

from apex.parallel import DistributedDataParallel as DDP
import torch

from asr.evaluation.utils import inference_model_from_checkpoint
from asr.evaluation.generator import Generator, GreedyGenerator, BeamGenerator
from asr.helpers import process_evaluation_batch, process_evaluation_epoch
from asr.models import EncoderDecoderModel, EncoderResult, DecoderResult, ModelResult
from asr.yt_dataset import YtTestDataset

from common.decoder import Decoder, create_decoder
from common.train_utils import nccl_barrier_on_cpu
from common.utils import num_gpus, gpu_id, Timer

LOG = logging.getLogger()


def main(local_rank: int,
         test_table: str,
         model_path: str,
         params: dict,
         tag: str,
         records_path: str,
         recognitions_path: str):
    assert (torch.cuda.is_available())

    LOG.info("loading model")

    model, features_config, dictionary = inference_model_from_checkpoint(model_path)
    decoder = create_decoder(nn_decoder=params.get("nn_decoder"),
                             text_decoder=params.get("text_decoder"),
                             dictionary=dictionary)

    if local_rank is not None:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    multi_gpu = torch.distributed.is_initialized()

    model.eval()
    model.cuda()
    model.encoder = DDP(model.encoder) if multi_gpu else model.encoder
    model.decoder = DDP(model.decoder) if multi_gpu else model.decoder

    if multi_gpu:
        LOG.debug("DISTRIBUTED EVALUATION with {} gpus".format(torch.distributed.get_world_size()))

    read_threads = params.get("read_threads")
    read_threads_per_gpu = (read_threads + num_gpus() - 1) // num_gpus()
    if read_threads_per_gpu * num_gpus() > read_threads:
        LOG.warning("Read threads per GPU is always rounded up, will use more threads than specified: "
                    "read threads = {}; num gpus = {}, will read in {} threads"
                    .format(read_threads,
                            num_gpus(),
                            read_threads_per_gpu * num_gpus()))

    LOG.info("loading test table")
    test_dataset = YtTestDataset(path=test_table,
                                 dictionary=dictionary,
                                 batch_size=params.get("batch_size"),
                                 features_config=features_config,
                                 sort_by_duration=params.get("sort_test_data"),
                                 in_memory=params.get("in_memory_test_data"),
                                 latin_policy=params.get("latin_policy"),
                                 read_threads=read_threads_per_gpu,
                                 pad_to=params.get("pad_to"))

    if model._model_context["transformer_decoder"]:
        if params.get("generator") == "greedy":
            generator = GreedyGenerator(model, dictionary)
        elif params.get("generator") == "beam_search":
            generator = BeamGenerator(model, dictionary, params.get("beam_size"))
        else:
            raise Exception(f"Unknown generator: {params.get('generator')}")
    else:
        generator = None

    LOG.info("evaluation start")
    transcripts, predictions = eval(test_dataset, model, generator, decoder)

    LOG.info("saving result")
    # save result from all gpus
    if multi_gpu:
        result = []
        for transcript, prediction in zip(transcripts, predictions):
            result.append({f"reference": transcript, "prediction": prediction})
        with open(f"result_{gpu_id()}.json", "w") as f:
            json.dump(result, f)

    if gpu_id() == 0:
        nccl_barrier_on_cpu()
        result = []
        # merge result from all gpus
        if multi_gpu:
            for i in range(torch.distributed.get_world_size()):
                with open(f"result_{i}.json", "r") as f:
                    result.extend(json.load(f))
                os.remove(f"result_{i}.json")
        else:
            for transcript, prediction in zip(transcripts, predictions):
                result.append({f"reference": transcript, "prediction": prediction})

        records = []
        recognitions = {}
        for i, item in enumerate(result):
            records.append({
                "id": f"#{i}_{tag}",
                "tags": [tag],
                "ref": item["reference"]
            })
            recognitions[f"#{i}_{tag}"] = item["prediction"]

        with open(records_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        with open(recognitions_path, "w") as f:
            json.dump(recognitions, f, indent=2, ensure_ascii=False)


def eval(test_dataset: YtTestDataset, model: EncoderDecoderModel, generator: Generator, decoder: Decoder):
    timer = Timer()

    with torch.no_grad():
        _global_var_dict = {
            "loss": 0,
            "predictions": [],
            "transcripts": [],
        }

        num_batches = 0
        for batch in test_dataset:
            if generator is not None:
                assert batch.features.size(0) == 1
                prediction = generator(batch.features)
                predictions = torch.LongTensor(prediction).unsqueeze(0).to(batch.features.device)
            else:
                encoder_result: EncoderResult = model.encoder(batch.features, None)
                decoder_result: DecoderResult = model.decoder(encoder_result, None, None)
                result = ModelResult(decoder_result.output, None, None)
                predictions = decoder.decode_probs(result.log_probs)

            loss = torch.tensor(0.).to(batch.features.device)
            process_evaluation_batch(loss, predictions, batch.tokens, batch.tokens_lengths, _global_var_dict, decoder)

            num_batches += 1
            sys.stderr.write(f"batch {num_batches} on worker {gpu_id()}\n")

        total_loss, total_wer = process_evaluation_epoch(_global_var_dict)
        LOG.info(f"==========>>>>>>Evaluation WER: {total_wer}\n")

    LOG.info(f"Evaluation time: {timer.passed()} seconds")

    return _global_var_dict["transcripts"], _global_var_dict["predictions"]
