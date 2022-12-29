# Если что здесь помойка которую я 10 лет не открывала и не использовала

from asr.disk_dataset import SpeechBatch
from asr.evaluation import GreedyGenerator, BeamGenerator

from asr.train import compute_loss
from asr.helpers import eval_quality
from asr.models import ModelResult
from common.disk_utils import BatchBlock
from asr.metrics import word_error_rate 

from torch.cuda import amp
import warnings
from scipy.io import wavfile
import torch

from tqdm.notebook import tqdm


def construct_references(decoder, tokens, tokens_lengths):
    targets_cpu = tokens.long().cpu()
    lengths_cpu = tokens_lengths.long().cpu()
    references = []
    for i in range(targets_cpu.size(0)):
        length = lengths_cpu[i].item()
        target = targets_cpu[i][:length].numpy().tolist()
        reference = decoder.dictionary().decode(target)
        references.append(reference)
    return references


def test(dataset, decoder, model, dictionary, decode_with_known_tokens=True, generator='beam'):
    max_speakers_num = dataset._max_speakers_num
    generator = BeamGenerator(model, dictionary, 10) if generator == 'beam' else \
                GreedyGenerator(model, dictionary) if generator == 'greedy' else None
    all_hypotheses, all_references = [], []
    with torch.no_grad():
        for batchblock in dataset:
            for speechbatch in batchblock:
                tokens = speechbatch.tokens
                tokens_lengths = speechbatch.tokens_lengths
                if torch.cuda.is_available():
                    speechbatch.cuda()
                if not decode_with_known_tokens:
                    hypotheses = [[] for _ in range(max_speakers_num)]
                    for f in speechbatch.features:
                        predictions = generator(f.reshape(1, f.shape[0], f.shape[1]))
                        for s, prediction in enumerate(predictions):
                            hypotheses[s].append(decoder.dictionary().decode(prediction))
                    references = [construct_references(decoder, t, t_len) for t, t_len in zip(tokens, tokens_lengths)]
                    wer, _, _ = word_error_rate([h for hyp in hypotheses for h in hyp],
                                                [r for ref in references for r in ref])
                else:
                    with amp.autocast():
                        result: ModelResult = model(speechbatch)
                    predictions = [decoder.decode_probs(log_probs) for log_probs in result.log_probs]
                    wer, hypotheses, references = eval_quality(predictions, tokens, tokens_lengths, decoder)
                print("wer =", wer)
                print()
                all_hypotheses += [h for hyp in hypotheses for h in hyp]
                all_references += [r for ref in references for r in ref]
                for i in range(len(hypotheses[0])):
                    for s in range(max_speakers_num):
                        print(f"Hyp{s + 1}: {hypotheses[s][i]}")
                        print(f"Ref{s + 1}: {references[s][i]}")
                    print()
            wer, _, _ = word_error_rate(all_hypotheses, all_references)
            print('Total WER:', wer)
            break


def test_single_with_audio(dataset, decoder, model, dictionary, decode_with_known_tokens=True, generator='beam'):
    max_speakers_num = dataset._max_speakers_num
    generator = BeamGenerator(model, dictionary, 10) if generator == 'beam' else \
                GreedyGenerator(model, dictionary) if generator == 'greedy' else None
    with torch.no_grad():
        mixed_audio, _, batch_block = iter(dataset).next_single_audio()

        for speechbatch in batch_block:
            tokens = speechbatch.tokens
            tokens_lengths = speechbatch.tokens_lengths
            if torch.cuda.is_available():
                speechbatch.cuda()
            if not decode_with_known_tokens:
                hypotheses = [[] for _ in range(max_speakers_num)]
                for f in speechbatch.features:
                    predictions = generator(f.reshape(1, f.shape[0], f.shape[1]))
                    for s, pred in enumerate(predictions):
                        hypotheses[s].append(decoder.dictionary().decode(pred))
                references = [construct_references(decoder, t, t_lens)
                              for t, t_lens in zip(tokens, tokens_lengths)]
                wer, _, _ = word_error_rate([h for hyp in hypotheses for h in hyp],
                                            [r for ref in references for r in ref])
            else:
                with amp.autocast():
                    result: ModelResult = model(speechbatch)
                predictions = [decoder.decode_probs(l) for l in result.log_probs]
                wer, hypotheses, references = eval_quality(predictions, tokens, tokens_lengths, decoder)
            print("wer =", wer)
            print()
            for s in range(max_speakers_num):
                print(f"Hyp{s + 1}: {hypotheses[s][0]}")
                print(f"Ref{s + 1}: {references[s][0]}")
            print()
        return mixed_audio


def test_my_dataset(decoder, model, dictionary, frame_shift, pad_to, batch_size,
                    parser, files, texts, generator='beam', decode_with_known_tokens=False):
    generator = BeamGenerator(model, dictionary, 10) if generator == 'beam' else \
                GreedyGenerator(model, dictionary) if generator == 'greedy' else None
    samples = []
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for file, text in zip(files, texts):
                rate, data = wavfile.read(file)
                data = data[:, 0]
                samples.append(parser.parse_from_audios(data, text))

    batch_block = BatchBlock([SpeechBatch.create_from_list(samples[i:i + batch_size], frame_shift, pad_to)
                              for i in range(0, len(samples), batch_size)], None)
    all_hypotheses, all_references = None, None
    for speechbatch in batch_block:
        max_speakers_num = len(speechbatch.tokens)
        if all_hypotheses is None or all_references is None:
            all_hypotheses = [[] for _ in range(max_speakers_num)]
            all_references = [[] for _ in range(max_speakers_num)]
        tokens = speechbatch.tokens
        tokens_lengths = speechbatch.tokens_lengths
        if torch.cuda.is_available():
            speechbatch.cuda()
            
        if decode_with_known_tokens:
            with amp.autocast():
                 result: ModelResult = model(speechbatch)

            predictions = [decoder.decode_probs(result.log_probs) for l in result.log_probs]
            wer, hypotheses, references = eval_quality(predictions, tokens, tokens_lengths, decoder)
        else:
            hypotheses = [[] for s in range(max_speakers_num)]
            print('Generating outputs...')
            for f in tqdm(speechbatch.features):
                predictions = generator(f.reshape(1, f.shape[0], f.shape[1]))
                for s in range(max_speakers_num):
                    hypotheses[s].append(decoder.dictionary().decode(predictions[s]))
            references = [construct_references(decoder, tokens[s], tokens_lengths[s])
                          for s in range(max_speakers_num)]

        for s in range(max_speakers_num):
            all_hypotheses[s] += hypotheses[s]
            all_references[s] += references[s]

    wer, _, _ = word_error_rate([h for hyp in all_hypotheses for h in hyp],
                                [r for ref in all_references for r in ref])
    return wer, all_hypotheses, all_references