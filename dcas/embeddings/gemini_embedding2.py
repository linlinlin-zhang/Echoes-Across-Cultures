from __future__ import annotations

import base64
import io
import json
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency at runtime
    genai = None
    genai_types = None

try:
    import torchaudio
except Exception:  # pragma: no cover - optional dependency at runtime
    torchaudio = None


@dataclass(frozen=True)
class GeminiEmbedding2Config:
    model_id: str = "gemini-embedding-2-preview"
    api_key: str | None = None
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    vertexai: bool = False
    vertex_project: str | None = None
    vertex_location: str | None = None
    output_dimensionality: int = 768
    task_type: str | None = None
    title: str | None = None
    max_seconds: float | None = 30.0
    target_sample_rate: int = 16_000
    window_count: int = 1
    window_strategy: str = "single"
    window_aggregate: str = "mean"
    request_timeout_s: int = 180
    max_retries: int = 5
    retry_backoff_s: float = 2.0
    audio_mime_type: str = "audio/wav"


def _wav_bytes_from_tensor(wav: torch.Tensor, sample_rate: int) -> bytes:
    if wav.ndim != 1:
        raise ValueError("expected mono waveform tensor")
    x = wav.detach().cpu().clamp(-1.0, 1.0)
    pcm16 = (x.numpy() * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(pcm16.tobytes())
    return buf.getvalue()


class GeminiEmbedding2Embedder:
    def __init__(self, cfg: GeminiEmbedding2Config):
        if torchaudio is None:
            raise ImportError("torchaudio is required for Gemini embedding extraction")
        api_key = cfg.api_key or os.environ.get("GEMINI_API_KEY", "")
        if api_key.strip() == "":
            raise RuntimeError("GEMINI_API_KEY is required for live Gemini embedding requests")
        self.cfg = GeminiEmbedding2Config(**{**cfg.__dict__, "api_key": api_key})
        self.session = None
        self.vertex_client = None
        if self.cfg.vertexai:
            if genai is None or genai_types is None:
                raise ImportError("google-genai is required for Vertex AI Gemini embedding requests")
            self.vertex_client = genai.Client(
                vertexai=True,
                api_key=api_key,
                project=self.cfg.vertex_project,
                location=self.cfg.vertex_location,
            )
        else:
            self.session = requests.Session()
            self.session.headers.update({"Content-Type": "application/json"})

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

    def _load_audio(self, path: str | Path, frame_offset: int = 0, num_frames: int | None = None) -> tuple[torch.Tensor, int]:
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
        wav = wav.mean(dim=0)
        if self.cfg.max_seconds is not None and float(self.cfg.max_seconds) > 0:
            max_len = int(float(self.cfg.max_seconds) * int(sr))
            if wav.shape[0] > max_len:
                wav = wav[:max_len]
        target_sr = int(self.cfg.target_sample_rate)
        if int(sr) != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=target_sr)
            sr = target_sr
        return wav, int(sr)

    def _prepare_window(self, path: str | Path, frame_offset: int = 0, num_frames: int | None = None) -> tuple[bytes, dict[str, Any]]:
        wav, sr = self._load_audio(path=path, frame_offset=int(frame_offset), num_frames=num_frames)
        audio_bytes = _wav_bytes_from_tensor(wav=wav, sample_rate=sr)
        meta = {
            "sample_rate": int(sr),
            "n_samples": int(wav.shape[0]),
            "duration_seconds": float(wav.shape[0]) / float(sr),
            "mime_type": self.cfg.audio_mime_type,
            "payload_bytes": int(len(audio_bytes)),
        }
        return audio_bytes, meta

    def prepare_file_report(self, path: str | Path) -> dict[str, Any]:
        plans = self._window_plan(path)
        windows: list[dict[str, Any]] = []
        total_payload = 0
        for i, (frame_offset, num_frames) in enumerate(plans):
            audio_bytes, meta = self._prepare_window(path=path, frame_offset=int(frame_offset), num_frames=num_frames)
            meta = dict(meta)
            meta["window_index"] = int(i)
            meta["frame_offset"] = int(frame_offset)
            meta["frame_offset_seconds"] = float(frame_offset) / float(meta["sample_rate"])
            windows.append(meta)
            total_payload += int(len(audio_bytes))
        return {
            "window_count": int(len(windows)),
            "window_strategy": str(self.cfg.window_strategy),
            "window_aggregate": str(self.cfg.window_aggregate),
            "total_payload_bytes": int(total_payload),
            "windows": windows,
        }

    def _request_body(self, audio_bytes: bytes, title: str | None = None) -> dict[str, Any]:
        req: dict[str, Any] = {
            "content": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": self.cfg.audio_mime_type,
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    }
                ]
            },
            "outputDimensionality": int(self.cfg.output_dimensionality),
        }
        if self.cfg.task_type:
            req["taskType"] = str(self.cfg.task_type)
        effective_title = title or self.cfg.title
        if effective_title:
            req["title"] = str(effective_title)
        return req

    def embed_audio_bytes(self, audio_bytes: bytes, title: str | None = None) -> np.ndarray:
        if self.cfg.vertexai:
            assert self.vertex_client is not None
            response = self.vertex_client.models.embed_content(
                model=self.cfg.model_id,
                contents=[
                    genai_types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=self.cfg.audio_mime_type,
                    )
                ],
                config={
                    "output_dimensionality": int(self.cfg.output_dimensionality),
                    **({"task_type": str(self.cfg.task_type)} if self.cfg.task_type else {}),
                    **({"title": str(title or self.cfg.title)} if (title or self.cfg.title) else {}),
                },
            )
            embeddings = getattr(response, "embeddings", None)
            if embeddings:
                values = getattr(embeddings[0], "values", None)
                if values:
                    return np.asarray(values, dtype=np.float32)
            embedding = getattr(response, "embedding", None)
            if embedding is not None:
                values = getattr(embedding, "values", None)
                if values:
                    return np.asarray(values, dtype=np.float32)
            raise RuntimeError(f"unexpected Vertex Gemini embedding payload: {response!r}")

        endpoint = f"{self.cfg.api_base}/models/{self.cfg.model_id}:embedContent?key={self.cfg.api_key}"
        payload = self._request_body(audio_bytes=audio_bytes, title=title)
        last_error: Exception | None = None

        for attempt in range(1, int(self.cfg.max_retries) + 1):
            try:
                assert self.session is not None
                r = self.session.post(
                    endpoint,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=int(self.cfg.request_timeout_s),
                )
                if r.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"transient Gemini error {r.status_code}: {r.text[:500]}")
                if r.status_code in {400, 404} and "gemini-embedding-2" in self.cfg.model_id:
                    raise RuntimeError(
                        "Gemini Embedding 2 audio request failed on the Gemini API route. "
                        "Current google-genai SDK behavior indicates multimodal embeddings are served via Vertex AI. "
                        f"status={r.status_code} body={r.text[:500]}"
                    )
                if r.status_code >= 400:
                    raise RuntimeError(f"Gemini request failed status={r.status_code} body={r.text[:500]}")
                r.raise_for_status()
                data = r.json()
                emb = data.get("embedding") or {}
                values = emb.get("values")
                if not isinstance(values, list) or not values:
                    raise RuntimeError(f"unexpected Gemini embedding payload: {json.dumps(data)[:800]}")
                arr = np.asarray(values, dtype=np.float32)
                if arr.ndim != 1:
                    raise RuntimeError(f"unexpected embedding shape: {arr.shape}")
                return arr
            except Exception as e:  # pragma: no cover - runtime/network dependent
                last_error = e
                if attempt >= int(self.cfg.max_retries):
                    break
                time.sleep(float(self.cfg.retry_backoff_s) * attempt)
        raise RuntimeError(f"Gemini embedding request failed after retries: {last_error}") from last_error

    def embed_file(self, path: str | Path, title: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        plans = self._window_plan(path)
        prep_windows: list[dict[str, Any]] = []
        embs: list[np.ndarray] = []
        window_failures: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for i, (frame_offset, num_frames) in enumerate(plans):
            try:
                audio_bytes, prep = self._prepare_window(path=path, frame_offset=int(frame_offset), num_frames=num_frames)
                prep = dict(prep)
                prep["window_index"] = int(i)
                prep["frame_offset"] = int(frame_offset)
                prep["frame_offset_seconds"] = float(frame_offset) / float(prep["sample_rate"])
                prep_windows.append(prep)
                embs.append(self.embed_audio_bytes(audio_bytes=audio_bytes, title=title))
            except Exception as e:
                last_error = e
                window_failures.append(
                    {
                        "window_index": int(i),
                        "frame_offset": int(frame_offset),
                        "num_frames": int(num_frames) if num_frames is not None else None,
                        "error": str(e),
                    }
                )
                continue

        if not embs:
            raise RuntimeError(f"all embedding windows failed for {Path(path)}: {last_error}") from last_error

        if len(embs) == 1:
            emb = embs[0].astype(np.float32)
        else:
            emb = np.stack(embs, axis=0).astype(np.float32).mean(axis=0).astype(np.float32)

        prep_report = {
            "window_count": int(len(prep_windows)),
            "window_strategy": str(self.cfg.window_strategy),
            "window_aggregate": str(self.cfg.window_aggregate),
            "windows": prep_windows,
            "embedding_dim": int(emb.shape[0]),
            "payload_bytes": int(sum(int(x["payload_bytes"]) for x in prep_windows)),
            "duration_seconds": float(sum(float(x["duration_seconds"]) for x in prep_windows) / max(1, len(prep_windows))),
            "sample_rate": int(prep_windows[0]["sample_rate"]) if prep_windows else int(self.cfg.target_sample_rate),
            "n_samples": int(sum(int(x["n_samples"]) for x in prep_windows)),
            "mime_type": str(prep_windows[0]["mime_type"]) if prep_windows else self.cfg.audio_mime_type,
            "window_failures": window_failures,
        }
        return emb, prep_report
