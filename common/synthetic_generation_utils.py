import math

import numpy as np

from typing import Callable, Tuple, List, Optional


def mix_audios(audios, rate, constant_gap: Optional[float], min_offset_seconds=1):
    if len(audios) > 1 and min_offset_seconds * rate >= min(len(w) for w in audios[:-1]):
        return None  # Can't mix audios

    if constant_gap is None:
        offsets = [np.random.randint(min_offset_seconds * rate, len(w)) for w in audios[:-1]]
    else:
        offsets = [int((1.0 - constant_gap) * len(w)) for w in audios[:-1]]
    silence_offset_lengths = np.cumsum([0] + offsets)

    # Add silence offset before speech
    audios = [np.concatenate((np.zeros(offset_len), a)) if offset_len != 0 else a
              for offset_len, a in zip(silence_offset_lengths, audios)]
    total_audio_len = max(len(w) for w in audios)

    # Add silence offset after speech
    audios = np.array([np.concatenate((w, np.zeros(total_audio_len - len(w)))) for w in audios])

    audios = audios.mean(axis=0)
    audios = normalize(audios)

    return audios


def normalize(a: np.ndarray):
    low = np.quantile(a, 0.05)
    high = np.quantile(a, 0.95)
    if high - low < 1e-8:
        return a - low
    a = (a - low) / (high - low)
    return a
