import logging
import random
import sys
import time

import numpy as np

import torch

LOG = logging.getLogger()


def log_debug_stdout(rank):
    if rank is None or rank == 0:
        logging.basicConfig(level=logging.DEBUG,
                            stream=sys.stdout,
                            format="%(levelname)s: %(asctime)s %(filename)s:%(lineno)d     %(message)s")


class Timer:
    def __init__(self):
        self._last_reset = time.time()

    def reset(self):
        self._last_reset = time.time()

    def passed(self):
        return time.time() - self._last_reset


def num_gpus():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def gpu_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


class TrainParams:
    def __init__(self, params: dict):
        self._params = params

    def get_param(self, key, default_value=None):
        return self._params.get(key, default_value)

    def save(self):
        return self._params

    def restore(self, params_dict):
        self._params = params_dict


def set_epoch_seed(seed, epoch):
    seed = seed if seed > 0 else -seed
    seed = seed + 1000 * epoch
    seed = seed % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = "\t%" + str(maxLen) + "s : %s"
    print("Arguments:")
    for keyPair in sorted(d.items()):
        print(fmtString % keyPair)


def exit_with_error(message, code=1):
    sys.stderr.write("Error: {__message}\n".format(__message=message))
    sys.exit(code)


def get_attention_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def get_padding_mask(lengths, max_length=None):
    if max_length is None:
        max_length = torch.max(lengths).item()
    batch_size = lengths.size(0)
    lens_arange = torch.arange(
        max_length
    ).to(  # move to the right device
        lengths.device
    ).view(  # reshape to (1, T)-shaped tensor
        1, max_length
    ).expand(
        batch_size, -1
    )
    if len(lengths.shape) == 2:
        lens_arange = torch.stack((lens_arange, lens_arange), 1)
        lengths_view = lengths.view(  # expand to (B, T)-shaped tensor
            batch_size, 2, 1
        ).expand(
            -1, -1, max_length
        )
    else:
        lengths_view = lengths.view(  # expand to (B, T)-shaped tensor
            batch_size, 1
        ).expand(
            -1, max_length
        )
    return lens_arange >= lengths_view


if __name__ == '__main__':
    a = torch.IntTensor([[1, 3], [2, 4], [3, 1]])
    print(get_padding_mask(a))
