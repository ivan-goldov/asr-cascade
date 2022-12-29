import math

import io

import librosa.feature
import numpy as np

import logging

import torch
import torch.nn as nn

LOG = logging.getLogger()


class PreprocessedFeaturesExtractor:
    def __init__(self, params, wave_augmentator, spec_augmentator):
        if params is not None and "num-mel-bins" in params:
            self.__features_dim = int(params["num-mel-bins"])
        else:
            self.__features_dim = None
        # TODO: dirty way, make it proper
        x = "fake"
        if wave_augmentator(x) != x:
            raise Exception("Wave augmentation is not supported for precomputed features")
        self._spec_augmentator = spec_augmentator

    def extract(self, yt_row):
        return self._spec_augmentator(np.load(io.BytesIO(yt_row[b"fbank"])))

    @classmethod
    def create(cls, config, augmentator, spec_augm):
        return PreprocessedFeaturesExtractor(config, augmentator, spec_augm)


def calculate_melspectrodram(data, sr, params):
    
#     win_length = int(sr * params['frame-length'] * 0.001) # frame size
#     hop_length = int(sr * params['frame-shift'] * 0.001)
#     n_fft = n_fft or 2 ** math.ceil(math.log2(win_length))

#     self.normalize = normalize
#     self.log = log
#     #TORCHSCRIPT: Check whether or not we need this
#     self.dither = dither
#     self.frame_splicing = frame_splicing
#     self.nfilt = nfilt
#     self.preemph = preemph
#     self.pad_to = pad_to
#     highfreq = highfreq or sample_rate / 2
#     window_fn = torch_windows.get(window, None)
#     window_tensor = window_fn(self.win_length,
#                               periodic=False) if window_fn else None
#     filterbanks = torch.tensor(
#         librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq,
#                                                 fmax=highfreq), dtype=torch.float).unsqueeze(0)
    
    return librosa.feature.melspectrogram(y=data, 
        sr=sr, 
        win_length=int(sr * params['frame-length'] * 0.001), 
        hop_length=int(sr * params['frame-shift'] * 0.001),
        window=params['window-type'],
        n_mels=params['num-mel-bins']
    ).transpose(1, 0)

constant=1e-5
class FilterbankFeatures(nn.Module):
    # For JIT. See https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length", "center", "log", "frame_splicing", "window", "normalize", "pad_to", "max_duration", "max_length"]

    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                       window="hamming", normalize="per_feature", n_fft=None,
                       preemph=0.97,
                       nfilt=64, lowfreq=0, highfreq=None, log=True, dither=constant,
                       pad_to=8,
                       max_duration=16.7,
                       frame_splicing=1):
        super(FilterbankFeatures, self).__init__()
        
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.win_length = int(sample_rate * window_size) # frame size
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        #TORCHSCRIPT: Check whether or not we need this
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq,
                                                    fmax=highfreq), dtype=torch.float).unsqueeze(0)
        # self.fb = filterbanks
        # self.window = window_tensor
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)
        # Calculate maximum sequence length (# frames)
        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.int)

    # do stft
    # TORCHSCRIPT: center removed due to bug
    def  stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window.to(dtype=torch.float))
    def forward(self, x, seq_len):
        
        dtype = x.dtype

        seq_len = self.get_seq_len(seq_len)
        
        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                                        dim=1)
            
        x  = self.stft(x)
            
            # get power spectrum
        x = x.pow(2).sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        # x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=seq_len.dtype).to(x.device).expand(x.size(0),
                                                                              max_len) >= seq_len.unsqueeze(1)

        x = x.masked_fill(mask.unsqueeze(1), 0)
        # TORCHSCRIPT: Is this del important? It breaks scripting
        # del mask
        # TORCHSCRIPT: Cant have mixed types. Using pad_to < 0 for "max"
        if self.pad_to < 0:
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)))
        elif self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            #            if pad_amt != 0:
            x = nn.functional.pad(x, (0, self.pad_to - pad_amt))
        
        return x # .to(dtype)

    @classmethod
    def from_config(cls, cfg, log=False):
        return cls(sample_rate=cfg['sample_rate'], window_size=cfg['window_size'],
                   window_stride=cfg['window_stride'], n_fft=cfg['n_fft'],
                   nfilt=cfg['features'], window=cfg['window'],
                   normalize=cfg['normalize'],
                   max_duration=cfg.get('max_duration', 16.7),
                   dither=cfg['dither'], pad_to=cfg.get("pad_to", 0),
                   frame_splicing=cfg.get("frame_splicing", 1), log=log)

class KaldiFeaturesExtractor:
    def __init__(self, params: dict, wave_augmentator, spec_augmentator):
        self._params = dict(params)
        self._mel_features_calcer = None
        if "normalize" in params:
            self._normalize = params.get("normalize")
            params.pop("normalize")
        else:
            self._normalize = False
        
#         self.__kaldi_features_calcer = FilterbankFeatures(
#             sample_rate=params['sample-frequency'], 
#             window_size=(params['frame-length'] / 1000),
#             window_stride=(params['frame-shift'] / 1000),
#             window=params['window-type'],  
#             preemph=params['preemphasis-coefficient'],
#             nfilt=params['num-mel-bins'],
#             lowfreq=params['low-freq'],
#             highfreq=params['high-freq'],
#             log=params['use-log-fbank'],
#             dither=params['dither']
#         ) #calculate_melspectrodram #, **params)
#         self._reference_amp = 1e-5 * (10 ** 4)
        self._wave_augmentator = wave_augmentator
        self._spec_augmentator = spec_augmentator
        self._sample_rate = params["sample-frequency"]

    def extract_raw(self, wave): # data):
        if self._mel_features_calcer is None:
            import kaldi_features
            self._mel_features_calcer = kaldi_features.KaldiFeaturesCalcer(self._params)
#         print('All data:', wave)
        sr_, wave_ = wave
#         print('Wave:', wave_.shape)
        if wave_ is None:
            return None
        if not isinstance(wave_, np.ndarray):
            raise ValueError
        if sr_ != self._sample_rate:
            raise ValueError
#         print('Wave:', wave_.shape)
        wave_ = self._wave_augmentator(wave_.astype(np.float32))
        wave_ = torch.tensor([wave_])
#         print('Wave:', wave_)
        seq_len = torch.tensor([wave_.shape[-1]])
#         result = self.__kaldi_features_calcer(wave_, seq_len) #(data=wave_, sr=sr_, params=self._params)
#         # print('mean MelFreq = {}'.format(' '.join(str(a) for a in list(np.array(result[0]).mean(axis=1)))), flush=True)
#         result = result.numpy().squeeze(0).transpose(1, 0)
        assert sr_ == self._sample_rate
#         print('Wave:', wave_[0].shape)
        result = self._mel_features_calcer.compute(wave_[0])
#         print('Result:', result)
        if self._normalize:
            m = np.mean(result)
            sd = np.std(result)
            result = (result - m) / (sd + 1e-9)
        result = self._spec_augmentator(result)

        return result

    def extract(self, record):
        return self.extract_raw((record["sample_rate"], record["data"]))

    @classmethod
    def create(cls, config, wave_augmentator, spec_augmentator):
        return KaldiFeaturesExtractor(config, wave_augmentator, spec_augmentator)


class FeatureExtractorFactory:
    extractors = {
        "disk-raw": KaldiFeaturesExtractor,
        "yt-raw": KaldiFeaturesExtractor,
        "example": KaldiFeaturesExtractor,
        "yt-features": PreprocessedFeaturesExtractor
    }

    def __init__(self):
        pass

    @classmethod
    def create(cls, data_type, config, wave_augmentator, spec_augmentator):
        impl = cls.extractors[data_type]
        return impl.create(config, wave_augmentator, spec_augmentator)
