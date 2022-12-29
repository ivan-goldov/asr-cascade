import logging
from typing import List

import torch
from torch import Tensor
import torch.distributed as dist

from asr.metrics import speaker_independent_word_error_rate, calculate_bleu, word_error_rate
from common.decoder import Decoder
from common.train_utils import all_reduce

LOG = logging.getLogger()


def eval_quality(predictions: List[Tensor], targets: List[Tensor], lengths: List[Tensor], decoder: Decoder):
    speakers_num = len(predictions)
    max_len = max(pred.shape[1] for pred in predictions)
    batch_sz = predictions[0].shape[0]
    predictions = [_extend_to(pred, max_len) for pred in predictions]
    targets = [_extend_to(targ, max_len) for targ in targets]

    predictions = torch.cat(tuple(predictions), 0)
    targets = torch.cat(tuple(targets), 0)
    lengths = torch.cat(tuple(lengths), 0)

    references = []
    with torch.no_grad():
        targets_cpu = targets.long().cpu()
        lengths_cpu = lengths.long().cpu()
        for i in range(targets_cpu.size(0)):
            length = lengths_cpu[i].item()
            target = targets_cpu[i][:length].numpy().tolist()
            reference = decoder.dictionary().decode(target)
            references.append(reference)
        hypotheses = decoder.decode_predictions(predictions)
    wer, _, _ = word_error_rate(hypotheses, references)
    return wer, \
           [hypotheses[s * batch_sz:(s + 1) * batch_sz] for s in range(speakers_num)], \
           [references[s * batch_sz:(s + 1) * batch_sz] for s in range(speakers_num)]


def _extend_to(a, width):
    if a.shape[1] < width:
        return torch.cat((a, torch.zeros(a.shape[0], width - a.shape[1], device=a.device)), 1)
    return a


def monitor_asr_train_progress(predictions: List[Tensor], targets: List[Tensor], lengths: List[Tensor],
                               decoder: Decoder):
    wer, hypotheses, references = eval_quality(predictions, targets, lengths, decoder)
    LOG.info(f"Train batch WER: {wer}")
    for i in range(len(hypotheses[0])):
        for s in range(len(hypotheses)):
            LOG.info(f"   Reference{s + 1}:  {references[s][i]}")
            LOG.info(f"   Prediction{s + 1}: {hypotheses[s][i]}")
        LOG.info("")
    return wer


def gather_transcripts(references: Tensor, lengths: Tensor, decoder: Decoder) -> List[str]:
    result = []
    references = references.long().cpu()
    lengths = lengths.long().cpu()
    for reference, lengths in zip(references, lengths):
        reference = reference[:lengths].numpy().tolist()
        reference = decoder.dictionary().decode(reference)
        result.append(reference)
    return result


def process_evaluation_batch(loss: Tensor, predictions: List[Tensor], transcripts: List[Tensor],
                             transcript_lengths: List[Tensor],
                             global_vars: dict, decoder: Decoder):
    global_vars["loss"] += loss.cpu().item()
    predictions = [decoder.decode_predictions(pr) for pr in predictions]
    transcripts = [gather_transcripts(transcripts[s], transcript_lengths[s], decoder) for s in range(len(predictions))]
    for i in range(len(predictions[0])):
        global_vars["predictions"].append([])
        global_vars["transcripts"].append([])
        for s in range(len(predictions)):
            global_vars["predictions"][-1].append(predictions[s][i])
            global_vars["transcripts"][-1].append(transcripts[s][i])


def process_evaluation_epoch(global_vars: dict, verbose=False):
    loss = global_vars["loss"]
    hypotheses = global_vars["predictions"]
    references = global_vars["transcripts"]

    wer, scores, num_words = speaker_independent_word_error_rate(hypotheses=hypotheses, references=references)
    LOG.info('Evaluation results:')
    for i in range(0, len(hypotheses) if verbose else 1):
        for s in range(len(hypotheses[i])):
            LOG.info(f'  Hyp{s + 1}: {hypotheses[i][s]}')
            LOG.info(f'  Ref{s + 1}: {references[i][s]}')
            LOG.info('')
    bleu = None  # calculate_bleu(hypotheses=hypotheses, references=references)
    multi_gpu = torch.distributed.is_initialized()

    loss = all_reduce(loss)
    scores = all_reduce(scores)
    num_words = all_reduce(num_words)
    wer = scores * 1.0 / num_words if multi_gpu else wer

    return loss, wer, bleu


