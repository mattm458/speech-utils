import torchaudio


def get_mel_spectrogram(
    sample_rate: int = 22050,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    n_mels: int = 80,
    power: int = 1,
    mel_scale: str = "slaney",
    norm: str = "slaney",
) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=power,
        mel_scale=mel_scale,
        norm=norm,
    )
