import copy
import random

import numpy as np


class SpecAugment:
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """

    def __init__(self,
                 num_frame_regions=1,
                 num_feature_regions=1,
                 frame_width=10,
                 feature_width=8,
                 zero_prob=0.5):

        self._num_frame_regions = num_frame_regions
        self._num_feature_regions = num_feature_regions

        self._frame_width = frame_width
        self._feature_width = feature_width

        self._zero_prob = zero_prob

    def __call__(self, x):
        num_frames = x.shape[0]
        num_features = x.shape[1]

        for _ in range(self._num_frame_regions):
            width = random.randint(1, self._frame_width)
            frame_from = random.randint(0, num_frames - width)
            frame_to = frame_from + width
            # duplicate some feature or just mask it
            val = 0
            if random.uniform(0, 1) > self._zero_prob:
                val = x[random.randint(frame_from, frame_to - 1), random.randint(0, num_features - 1)]
            x[frame_from:frame_to, :] = val

        for _ in range(self._num_feature_regions):
            width = random.randint(1, self._feature_width)
            features_from = random.randint(0, num_features - width)
            features_to = features_from + width

            val = 0
            if random.uniform(0, 1) > self._zero_prob:
                val = x[random.randint(0, num_frames - 1), random.randint(features_from, features_to - 1)]

            x[:, features_from:features_to] = val

        # x = x.masked_fill(mask.to(device=x.device), 0)

        return x


class CloudPhoneAugmentation:
    def __init__(self,
                 alpha_from: float,
                 alpha_to: float,
                 height_from: int,
                 height_to: int,
                 width_from: float,
                 width_to: float,
                 fill_prob: float):
        self._alpha_from = alpha_from
        self._alpha_to = alpha_to
        self._height_from = height_from
        self._height_to = height_to
        self._width_from = width_from
        self._width_to = width_to
        self._fill_prob = fill_prob

    def __call__(self, x):
        alpha = random.uniform(self._alpha_from, self._alpha_to)
        offset = 0
        f_count = x.shape[1]
        frames = x.shape[0]
        should_fill = random.uniform(0, 1) < self._fill_prob
        x = np.transpose(np.copy(x))
        while offset < frames:
            height = random.randint(self._height_from, self._height_to)
            width = random.randint(self._width_from, self._width_to)
            next_offset = min(offset + width, frames)
            if should_fill:
                v = x[height, offset:next_offset]
                x[height:f_count, offset:next_offset] = v * alpha
            else:
                x[height:f_count, offset:next_offset] *= alpha
            offset = next_offset
        return np.transpose(x)


spec_augmentation_types = {
    "spec_augment": SpecAugment,
    "phone_aug": CloudPhoneAugmentation
}


class SpectrogramAugmentatator:
    """Spectrogram augmentation
    """

    def __init__(self, augmentations=None):
        # random here global, cause snapshots/etc
        self._rng = random.Random()
        self._pipeline = augmentations if augmentations is not None else []

    def __call__(self, spectogram):
        for (prob, p) in self._pipeline:
            if self._rng.random() < prob:
                spectogram = p(spectogram)
        return spectogram

    @classmethod
    def from_config(cls, augmentation_config):
        augmentations = []

        for augmentation_type, config in augmentation_config.items():
            if augmentation_type not in spec_augmentation_types:
                raise Exception("augmentation {} not known.".format(augmentation_type))
            impl = spec_augmentation_types[augmentation_type]
            config = copy.deepcopy(config)
            p = config["prob"]
            config.pop("prob")
            augmentations.append((p,
                                  impl(**config)))

        return cls(augmentations=augmentations)
