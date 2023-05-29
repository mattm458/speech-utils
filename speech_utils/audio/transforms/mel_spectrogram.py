import logging
import math
import os
from os import path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram


def mel_to_log_mel(mel: Tensor) -> Tensor:
    return torch.log(torch.clamp(mel, min=1e-5))


def log_mel_to_mel(log_mel: Tensor) -> Tensor:
    return torch.exp(log_mel)


def pad_wav_multiple(wav: Tensor, multiple: int = 256) -> Tuple[Tensor, int]:
    wav_len = wav.shape[-1]
    padded_len = math.ceil(wav_len / multiple) * multiple

    pad_len = padded_len - wav_len

    return F.pad(wav, (0, pad_len)), pad_len


__HIFI_GAN_PAD = (1024 - 256) // 2


def pad_mel(wav: Tensor):
    return F.pad(wav, (__HIFI_GAN_PAD, __HIFI_GAN_PAD))


class TacotronMelSpectrogram(MelSpectrogram):
    def __init__(self, n_mels=80, cache=False, cache_dir=None):
        super().__init__(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=8000.0,
            n_mels=n_mels,
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
            center=True,
        )

        assert (
            cache and cache_dir is not None
        ) or not cache, "If caching spectrograms, a cache directory is required"

        if cache and path.exists(cache_dir):
            logging.warn(
                f"Cache directory {cache_dir} already exists! Are you sure the cached spectrograms are correct?"
            )
        elif cache and not path.exists(cache_dir):
            os.mkdir(cache_dir)

        self.cache = cache
        self.cache_dir = cache_dir

    def __call__(self, x: Tensor, id: Optional[str] = None) -> Tensor:
        mel_spectrogram = None

        filepath = None
        uncached = False
        if self.cache and id is not None:
            filepath = path.join(self.cache_dir, f"{id}.pt")
            try:
                mel_spectrogram = torch.load(filepath)
            except:
                uncached = True

        if mel_spectrogram is None:
            mel_spectrogram = super().__call__(x).swapaxes(-1, -2)
            mel_spectrogram = mel_to_log_mel(mel_spectrogram)

            if uncached and filepath is not None:
                torch.save(mel_spectrogram, filepath)

        return mel_spectrogram


class HifiGanMelSpectrogram(MelSpectrogram):
    def __init__(self):
        super().__init__(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=None,
            n_mels=80,
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
            center=False,
        )

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x).swapaxes(-1, -2)
