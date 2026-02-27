from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass(frozen=True)
class WaveformStyleTransferResult:
    output_path: str
    sample_rate: int
    n_samples: int
    source_audio_path: str
    style_audio_path: str
    metrics: dict[str, float]


def _load_audio_mono(path: str | Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.ndim != 2:
        raise ValueError("audio tensor must be [channels, samples]")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=int(target_sr))
    return wav


def _align_style_to_source_length(source: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_src = int(source.shape[-1])
    n_sty = int(style.shape[-1])
    if n_src <= 0 or n_sty <= 0:
        raise ValueError("source/style waveform must be non-empty")
    if n_sty < n_src:
        reps = (n_src + n_sty - 1) // n_sty
        style = style.repeat(1, int(reps))[..., :n_src]
    elif n_sty > n_src:
        style = style[..., :n_src]
    return source[..., :n_src], style


def _spectral_centroid(mag: torch.Tensor, sample_rate: int, n_fft: int) -> float:
    # mag: [freq_bins, frames]
    if mag.numel() == 0:
        return 0.0
    freq_bins = mag.shape[0]
    freqs = torch.linspace(0.0, float(sample_rate) / 2.0, steps=freq_bins, device=mag.device, dtype=mag.dtype)
    num = (freqs[:, None] * mag).sum(dim=0)
    den = mag.sum(dim=0).clamp_min(1e-8)
    cent = num / den
    return float(cent.mean().item())


def transfer_waveform_style(
    source_audio_path: str | Path,
    style_audio_path: str | Path,
    output_wav_path: str | Path,
    alpha: float = 0.7,
    target_sr: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
    max_seconds: float | None = None,
    peak_norm: float = 0.98,
) -> WaveformStyleTransferResult:
    if n_fft < 128:
        raise ValueError("n_fft must be >= 128")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")
    if win_length is None:
        win_length = int(n_fft)
    if win_length <= 0 or win_length > n_fft:
        raise ValueError("win_length must be in (0, n_fft]")
    a = float(alpha)
    if a < 0.0 or a > 1.5:
        raise ValueError("alpha must be in [0.0, 1.5]")

    src_path = Path(source_audio_path)
    sty_path = Path(style_audio_path)
    if not src_path.exists():
        raise FileNotFoundError(f"source audio not found: {src_path}")
    if not sty_path.exists():
        raise FileNotFoundError(f"style audio not found: {sty_path}")

    src = _load_audio_mono(src_path, target_sr=int(target_sr))
    sty = _load_audio_mono(sty_path, target_sr=int(target_sr))

    if max_seconds is not None and float(max_seconds) > 0:
        n = int(round(float(max_seconds) * int(target_sr)))
        src = src[..., :n]
        sty = sty[..., :n]
    src, sty = _align_style_to_source_length(src, sty)

    window = torch.hann_window(int(win_length), dtype=src.dtype, device=src.device)

    src_spec = torch.stft(
        src.squeeze(0),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        return_complex=True,
    )
    sty_spec = torch.stft(
        sty.squeeze(0),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        return_complex=True,
    )

    src_mag = torch.abs(src_spec).clamp_min(1e-8)
    sty_mag = torch.abs(sty_spec).clamp_min(1e-8)
    src_phase = src_spec / src_mag

    src_log = torch.log(src_mag)
    sty_log = torch.log(sty_mag)

    src_mu = src_log.mean(dim=1, keepdim=True)
    src_std = src_log.std(dim=1, keepdim=True).clamp_min(1e-6)
    sty_mu = sty_log.mean(dim=1, keepdim=True)
    sty_std = sty_log.std(dim=1, keepdim=True).clamp_min(1e-6)

    transferred_log = ((src_log - src_mu) / src_std) * sty_std + sty_mu
    transferred_mag = torch.exp(transferred_log)
    out_mag = ((1.0 - a) * src_mag + a * transferred_mag).clamp_min(1e-8)
    out_spec = out_mag * src_phase

    out_wav = torch.istft(
        out_spec,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        length=int(src.shape[-1]),
    ).unsqueeze(0)

    peak = float(out_wav.abs().max().item())
    if peak > 0 and peak_norm > 0:
        out_wav = out_wav * (float(peak_norm) / peak)

    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), out_wav.cpu(), int(target_sr))

    out_spec_check = torch.stft(
        out_wav.squeeze(0),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window.cpu(),
        return_complex=True,
    )
    out_mag_check = torch.abs(out_spec_check).clamp_min(1e-8)

    src_cent = _spectral_centroid(src_mag, sample_rate=int(target_sr), n_fft=int(n_fft))
    sty_cent = _spectral_centroid(sty_mag, sample_rate=int(target_sr), n_fft=int(n_fft))
    out_cent = _spectral_centroid(out_mag_check, sample_rate=int(target_sr), n_fft=int(n_fft))
    src_to_sty = abs(src_cent - sty_cent)
    out_to_sty = abs(out_cent - sty_cent)
    src_to_out = abs(src_cent - out_cent)

    metrics = {
        "alpha": float(a),
        "spectral_centroid_src_hz": float(src_cent),
        "spectral_centroid_style_hz": float(sty_cent),
        "spectral_centroid_out_hz": float(out_cent),
        "distance_src_to_style_hz": float(src_to_sty),
        "distance_out_to_style_hz": float(out_to_sty),
        "distance_src_to_out_hz": float(src_to_out),
        "style_alignment_gain_hz": float(src_to_sty - out_to_sty),
        "output_peak": float(out_wav.abs().max().item()),
    }

    return WaveformStyleTransferResult(
        output_path=str(out_path),
        sample_rate=int(target_sr),
        n_samples=int(out_wav.shape[-1]),
        source_audio_path=str(src_path),
        style_audio_path=str(sty_path),
        metrics=metrics,
    )
