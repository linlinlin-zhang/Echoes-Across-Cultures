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
    layer_indices: list[int] | None = None
    layer_weights: list[float] | None = None
    max_seconds: float | None = 30.0
    window_count: int = 1
    window_strategy: str = "single"
    window_aggregate: str = "mean"
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
        if cfg.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be one of: mean, cls")
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

    def _pool_hidden(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            return hidden[:, 0, :]
        if mask is not None and (mask.ndim != 2 or mask.shape[1] != hidden.shape[1]):
            mask = None
        return _masked_mean(hidden=hidden, mask=mask)

    def _resolve_layer_indices(self, n_layers: int) -> list[int]:
        raw = self.cfg.layer_indices
        if not raw:
            return [n_layers - 1]
        resolved: list[int] = []
        for idx in raw:
            pos = int(idx)
            if pos < 0:
                pos += int(n_layers)
            if pos < 0 or pos >= int(n_layers):
                raise ValueError(f"layer index out of range: {idx} for n_layers={n_layers}")
            if pos not in resolved:
                resolved.append(pos)
        if not resolved:
            raise ValueError("layer_indices resolved to an empty selection")
        return resolved

    def _aggregate_layers(self, pooled_layers: list[torch.Tensor]) -> torch.Tensor:
        if len(pooled_layers) == 1:
            return pooled_layers[0]
        stack = torch.stack(pooled_layers, dim=0)
        weights = self.cfg.layer_weights
        if weights:
            if len(weights) != len(pooled_layers):
                raise ValueError(
                    f"layer_weights length {len(weights)} does not match selected layers {len(pooled_layers)}"
                )
            w = torch.tensor([float(x) for x in weights], dtype=stack.dtype, device=stack.device)
            denom = w.sum().clamp_min(1e-6)
            w = w / denom
            return (stack * w[:, None]).sum(dim=0)
        return stack.mean(dim=0)

    def _window_plan(self, path: str | Path) -> list[tuple[int, int | None]]:
        audio_path = Path(path)
        try:
            info = torchaudio.info(str(audio_path))
        except Exception:
            return [(0, None)]

        sr = int(info.sample_rate)
        total_frames = int(info.num_frames)
        if sr <= 0 or total_frames <= 0:
            return [(0, None)]

        count = max(1, int(self.cfg.window_count))
        if self.cfg.max_seconds is None or float(self.cfg.max_seconds) <= 0:
            return [(0, total_frames)]

        segment_frames = max(1, int(float(self.cfg.max_seconds) * sr))
        if total_frames <= segment_frames or count == 1 or str(self.cfg.window_strategy).strip().lower() == "single":
            return [(0, min(segment_frames, total_frames))]

        max_offset = max(0, total_frames - segment_frames)
        starts = np.linspace(0, max_offset, num=count, dtype=np.int64).tolist()
        unique_starts: list[int] = []
        seen: set[int] = set()
        for start in starts:
            s = int(start)
            if s not in seen:
                seen.add(s)
                unique_starts.append(s)
        return [(int(s), min(segment_frames, total_frames - int(s))) for s in unique_starts]

    def _load_audio(self, path: str | Path, frame_offset: int = 0, num_frames: int | None = None) -> torch.Tensor:
        audio_path = Path(path)
        load_kwargs: dict[str, int] = {}
        if int(frame_offset) > 0:
            load_kwargs["frame_offset"] = int(frame_offset)
        if num_frames is not None and int(num_frames) > 0:
            load_kwargs["num_frames"] = int(num_frames)
        elif self.cfg.max_seconds is not None and float(self.cfg.max_seconds) > 0:
            try:
                info = torchaudio.info(str(audio_path))
                sr_hint = int(info.sample_rate)
                if sr_hint > 0:
                    load_kwargs["num_frames"] = int(float(self.cfg.max_seconds) * sr_hint)
            except Exception:
                pass
        wav, sr = torchaudio.load(str(audio_path), **load_kwargs)
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
        need_hidden_states = bool(self.cfg.layer_indices)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=need_hidden_states)
            mask = inputs.get("attention_mask")
            if need_hidden_states:
                hidden_states = getattr(out, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError("model did not return hidden_states for layer aggregation")
                selected = self._resolve_layer_indices(n_layers=len(hidden_states))
                pooled_layers = [self._pool_hidden(hidden=hidden_states[idx], mask=mask).squeeze(0) for idx in selected]
                emb = self._aggregate_layers(pooled_layers).unsqueeze(0)
            else:
                emb = self._pool_hidden(hidden=out.last_hidden_state, mask=mask)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def embed_file(self, path: str | Path) -> np.ndarray:
        plans = self._window_plan(path)
        embs: list[np.ndarray] = []
        last_error: Exception | None = None
        for frame_offset, num_frames in plans:
            try:
                wav = self._load_audio(path=path, frame_offset=int(frame_offset), num_frames=num_frames)
                embs.append(self.embed_waveform(wav=wav, sampling_rate=self.sampling_rate))
            except Exception as e:
                last_error = e
                continue
        if not embs:
            raise RuntimeError(f"all embedding windows failed for {Path(path)}: {last_error}") from last_error
        if len(embs) == 1:
            return embs[0]
        stack = np.stack(embs, axis=0).astype(np.float32)
        return stack.mean(axis=0).astype(np.float32)
