import math
from typing import Tuple

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


class TacotronMelSpectrogram(MelSpectrogram):
    def __init__(self):
        super().__init__(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=8000.0,
            n_mels=80,
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
            center=False,
        )

    def __call__(self, x: Tensor) -> Tensor:
        mel_spectrogram = super().__call__(x).swapaxes(1, 2)
        return mel_to_log_mel(mel_spectrogram)


def HifiGanMelSpectrogram() -> MelSpectrogram:
    return MelSpectrogram(
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
