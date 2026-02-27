from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import torchaudio
except Exception:  # pragma: no cover - optional dependency at runtime
    torchaudio = None

try:
    from transformers import AutoFeatureExtractor, AutoModel
except Exception:  # pragma: no cover - optional dependency at runtime
    AutoFeatureExtractor = None
    AutoModel = None


@dataclass(frozen=True)
class CultureMERTConfig:
    model_id: str = "ntua-slp/CultureMERT-95M"
    device: str | None = None
    pooling: str = "mean"
    max_seconds: float | None = 30.0
    trust_remote_code: bool = True


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    m = mask.to(hidden.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1e-6)
    return (hidden * m).sum(dim=1) / denom


class CultureMERTEmbedder:
    def __init__(self, cfg: CultureMERTConfig):
        if AutoFeatureExtractor is None or AutoModel is None:
            raise ImportError("transformers is required for CultureMERT embedding extraction")
        if torchaudio is None:
            raise ImportError("torchaudio is required for CultureMERT embedding extraction")

        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            cfg.model_id,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            cfg.model_id,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.model.eval()
        self.model.to(self.device)

        sr = getattr(self.feature_extractor, "sampling_rate", None)
        self.sampling_rate = int(sr) if sr is not None else 24_000

    def _load_audio(self, path: str | Path) -> torch.Tensor:
        audio_path = Path(path)
        wav, sr = torchaudio.load(str(audio_path))
        if wav.ndim != 2:
            raise ValueError(f"invalid waveform shape for {audio_path}: {tuple(wav.shape)}")
        wav = wav.mean(dim=0)  # mixdown to mono
        if self.cfg.max_seconds is not None and float(self.cfg.max_seconds) > 0:
            max_len = int(float(self.cfg.max_seconds) * sr)
            if wav.shape[0] > max_len:
                wav = wav[:max_len]
        if int(sr) != int(self.sampling_rate):
            wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=int(self.sampling_rate))
        return wav

    def embed_waveform(self, wav: torch.Tensor, sampling_rate: int) -> np.ndarray:
        if wav.ndim != 1:
            raise ValueError("expected 1D mono waveform")
        x = wav
        sr = int(sampling_rate)
        if sr != int(self.sampling_rate):
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=int(self.sampling_rate))
            sr = int(self.sampling_rate)
        inputs = self.feature_extractor(
            x.detach().cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            hidden = out.last_hidden_state
            if self.cfg.pooling == "cls":
                emb = hidden[:, 0, :]
            else:
                emb = _masked_mean(hidden=hidden, mask=inputs.get("attention_mask"))
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def embed_file(self, path: str | Path) -> np.ndarray:
        wav = self._load_audio(path)
        return self.embed_waveform(wav=wav, sampling_rate=self.sampling_rate)

