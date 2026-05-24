from __future__ import annotations

import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dcas.data.interactions import Interaction
from dcas.data.npz_tracks import Tracks, load_tracks
from dcas.embeddings.culturemert import CultureMERTConfig, CultureMERTEmbedder
from dcas.ot.sinkhorn import sinkhorn_plan, squared_euclidean_cost
from dcas.serialization import load_checkpoint

from .paths import Storage


DEFAULT_TRACKS_REL = "public/merged/tracks_culturemert.npz"
DEFAULT_METADATA_REL = "public/merged/metadata_merged.csv"
DEFAULT_MODEL_REL = "models/dcas_full_v4_main_culturemert_stage3.pt"

LOW_SIGNAL_TERMS = (
    "karaoke",
    "backing track",
    "originally performed by",
    "tribute",
    "sound effect",
    "ringtone",
    "loop ",
    " loops",
)


@dataclass(frozen=True)
class MainlineWeights:
    relevance: float = 0.48
    novelty: float = 0.10
    target_affinity: float = 0.22
    minority: float = 0.14
    source: float = 0.06
    diversity_lambda: float = 0.03


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm_key(value: Any) -> str:
    text = _clean(value).lower()
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix in {".m4a", ".mp4", ".aac"}:
        return "audio/mp4"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"


def _metadata_audio_path(value: str, metadata_path: Path) -> Path:
    text = _clean(value)
    if text.startswith("/mnt/"):
        parts = text.split("/")
        if len(parts) >= 4 and len(parts[2]) == 1:
            drive_path = Path(f"{parts[2].upper()}:/").joinpath(*parts[3:])
            if drive_path.exists():
                return drive_path
    path = Path(text)
    if not path.is_absolute():
        path = (metadata_path.parent / path).resolve()
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


class MainlineRecommendationPlatform:
    """Cached service wrapper for the DCAS mainline recommender.

    The research benchmark functions are batch/evaluation oriented. This class
    keeps the same DCAS + OT + calibrated reranking ingredients, but exposes a
    seed-track API shape that can be used by the backend and, later, the web UI.
    """

    def __init__(
        self,
        storage: Storage,
        *,
        prefer_cuda: bool = False,
        tracks_rel: str = DEFAULT_TRACKS_REL,
        metadata_rel: str = DEFAULT_METADATA_REL,
        model_rel: str = DEFAULT_MODEL_REL,
    ) -> None:
        self.storage = storage
        self.prefer_cuda = bool(prefer_cuda)
        self.device = torch.device("cuda" if self.prefer_cuda and torch.cuda.is_available() else "cpu")
        self.tracks_path = storage.resolve_rel(tracks_rel)
        self.metadata_path = storage.resolve_rel(metadata_rel)
        self.model_path = storage.resolve_rel(model_rel)

        missing = [str(p) for p in [self.tracks_path, self.metadata_path, self.model_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"mainline platform input missing: {missing}")

        self.loaded_at = time.time()
        self.tracks: Tracks = load_tracks(str(self.tracks_path))
        self.track_id_to_idx = {str(tid): int(i) for i, tid in enumerate(self.tracks.track_id.tolist())}
        self.metadata_by_id, self.metadata_fields = self._load_metadata(self.metadata_path)
        self.culture_counts = Counter(str(c) for c in self.tracks.culture.tolist())
        self.source_counts = Counter(str(s) for s in (self.tracks.source_dataset.tolist() if self.tracks.source_dataset is not None else []))
        self.model, _ = load_checkpoint(str(self.model_path), map_location=str(self.device))
        self.model.eval()
        self.model.to(self.device)
        self.zs_all, self.za_all = self._encode_catalog()
        self.culture_names, self.culture_centroids = self._build_culture_centroids()
        self._culturemert_embedders: dict[tuple[Any, ...], CultureMERTEmbedder] = {}

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "loaded_at": self.loaded_at,
            "device": str(self.device),
            "tracks_path": str(self.tracks_path),
            "metadata_path": str(self.metadata_path),
            "model_path": str(self.model_path),
            "n_tracks": int(len(self.tracks)),
            "embedding_dim": int(self.tracks.dim),
            "n_metadata_rows": int(len(self.metadata_by_id)),
            "metadata_fields": list(self.metadata_fields),
            "cultures": dict(sorted(self.culture_counts.items())),
            "sources": dict(sorted(self.source_counts.items())),
            "model_cfg": {
                "in_dim": int(self.model.cfg.in_dim),
                "zc_dim": int(self.model.cfg.zc_dim),
                "zs_dim": int(self.model.cfg.zs_dim),
                "za_dim": int(self.model.cfg.za_dim),
                "n_cultures": int(self.model.cfg.n_cultures),
                "n_sources": int(self.model.cfg.n_sources),
            },
        }

    def track(self, track_id: str) -> dict[str, Any]:
        idx = self._require_track(track_id)
        return self._track_payload(idx)

    def cultures(self) -> dict[str, Any]:
        return {
            "ok": True,
            "cultures": [
                {"culture": name, "count": int(count)}
                for name, count in sorted(self.culture_counts.items())
            ],
            "sources": [
                {"source_dataset": name, "count": int(count)}
                for name, count in sorted(self.source_counts.items())
            ],
        }

    def catalog(
        self,
        *,
        culture: str | None = None,
        source_dataset: str | None = None,
        q: str | None = None,
        limit: int = 24,
        random_seed: int | None = 42,
        exclude_low_signal: bool = True,
    ) -> dict[str, Any]:
        limit = max(1, min(200, int(limit)))
        candidates = np.arange(len(self.tracks), dtype=np.int64)
        culture_key = _clean(culture)
        source_key = _clean(source_dataset)
        query = _norm_key(q)

        if culture_key:
            candidates = candidates[self.tracks.culture.astype(str)[candidates] == culture_key]
        if source_key and self.tracks.source_dataset is not None:
            candidates = candidates[self.tracks.source_dataset.astype(str)[candidates] == source_key]

        keep: list[int] = []
        query_terms = [term for term in query.split(" ") if term]
        for idx in candidates.tolist():
            idx = int(idx)
            if exclude_low_signal and self._is_low_signal(idx):
                continue
            if query_terms and not self._matches_query(idx, query_terms):
                continue
            keep.append(idx)

        if random_seed is not None:
            rng = np.random.default_rng(int(random_seed))
            rng.shuffle(keep)

        items = [self._track_payload(idx) for idx in keep[:limit]]
        return {
            "ok": True,
            "request": {
                "culture": culture_key,
                "source_dataset": source_key,
                "q": query,
                "limit": limit,
                "random_seed": random_seed,
                "exclude_low_signal": bool(exclude_low_signal),
            },
            "total_available": int(len(keep)),
            "items": items,
        }

    def random_track(
        self,
        *,
        culture: str | None = None,
        source_dataset: str | None = None,
        random_seed: int | None = 42,
        exclude_low_signal: bool = True,
    ) -> dict[str, Any]:
        result = self.catalog(
            culture=culture,
            source_dataset=source_dataset,
            limit=1,
            random_seed=random_seed,
            exclude_low_signal=exclude_low_signal,
        )
        items = result.get("items", [])
        if not items:
            raise ValueError("no matching track available")
        return {"ok": True, "track": items[0], "request": result["request"], "total_available": result["total_available"]}

    def embed_audio_file(
        self,
        path: str | Path,
        *,
        model_id: str = "ntua-slp/CultureMERT-95M",
        pooling: str = "mean",
        max_seconds: float | None = 30.0,
        window_count: int = 1,
        window_strategy: str = "single",
        window_aggregate: str = "mean",
    ) -> np.ndarray:
        max_seconds_key = None if max_seconds is None else float(max_seconds)
        key = (
            str(model_id),
            str(self.device),
            str(pooling),
            max_seconds_key,
            int(window_count),
            str(window_strategy),
            str(window_aggregate),
        )
        embedder = self._culturemert_embedders.get(key)
        if embedder is None:
            cfg = CultureMERTConfig(
                model_id=str(model_id),
                device=str(self.device),
                pooling=str(pooling),
                max_seconds=max_seconds,
                window_count=int(window_count),
                window_strategy=str(window_strategy),
                window_aggregate=str(window_aggregate),
            )
            embedder = CultureMERTEmbedder(cfg)
            self._culturemert_embedders[key] = embedder
        emb = embedder.embed_file(path)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if int(emb.shape[0]) != int(self.tracks.dim):
            raise ValueError(f"uploaded embedding dim={emb.shape[0]} does not match catalog dim={self.tracks.dim}")
        if not np.isfinite(emb).all():
            raise ValueError("uploaded embedding contains non-finite values")
        return emb.astype(np.float32)

    def recommend_audio_file(
        self,
        *,
        audio_path: str | Path,
        upload_info: dict[str, Any] | None = None,
        seed_culture: str | None = None,
        target_culture: str | None = None,
        mode: str = "open",
        k: int = 10,
        recall_k: int = 900,
        random_seed: int | None = 42,
        exclude_same_artist: bool = False,
        exclude_low_signal: bool = True,
        weights: MainlineWeights | None = None,
        culturemert_model_id: str = "ntua-slp/CultureMERT-95M",
        pooling: str = "mean",
        max_seconds: float | None = 30.0,
        window_count: int = 1,
        window_strategy: str = "single",
        window_aggregate: str = "mean",
    ) -> dict[str, Any]:
        started = time.time()
        emb = self.embed_audio_file(
            audio_path,
            model_id=culturemert_model_id,
            pooling=pooling,
            max_seconds=max_seconds,
            window_count=window_count,
            window_strategy=window_strategy,
            window_aggregate=window_aggregate,
        )
        result = self.recommend_embedding(
            embedding=emb,
            upload_info=upload_info or {},
            seed_culture=seed_culture,
            target_culture=target_culture,
            mode=mode,
            k=k,
            recall_k=recall_k,
            random_seed=random_seed,
            exclude_same_artist=exclude_same_artist,
            exclude_low_signal=exclude_low_signal,
            weights=weights,
        )
        result["embedding"] = {
            "model_id": str(culturemert_model_id),
            "pooling": str(pooling),
            "dim": int(emb.shape[0]),
            "max_seconds": max_seconds,
            "window_count": int(window_count),
            "window_strategy": str(window_strategy),
            "window_aggregate": str(window_aggregate),
            "elapsed_seconds": float(time.time() - started),
        }
        return result

    def recommend_embedding(
        self,
        *,
        embedding: np.ndarray,
        upload_info: dict[str, Any],
        seed_culture: str | None = None,
        target_culture: str | None = None,
        mode: str = "open",
        k: int = 10,
        recall_k: int = 900,
        random_seed: int | None = 42,
        exclude_same_artist: bool = False,
        exclude_low_signal: bool = True,
        weights: MainlineWeights | None = None,
    ) -> dict[str, Any]:
        weights = weights or MainlineWeights()
        mode = str(mode or "open").strip().lower()
        if mode not in {"open", "target"}:
            raise ValueError("mode must be one of: open, target")
        k = max(1, min(100, int(k)))
        recall_k = max(k, min(5000, int(recall_k)))

        emb, zs_seed, za_seed = self._encode_external_embedding(embedding)
        inferred = self._infer_culture(zs_seed)
        effective_target = _clean(target_culture) or _clean(seed_culture) or str(inferred["culture"])
        if effective_target not in self.culture_counts:
            raise ValueError(f"unknown target_culture={effective_target}")

        candidate_idx = self._candidate_indices(
            seed_idx=np.array([], dtype=np.int64),
            mode=mode,
            target_culture=effective_target,
            exclude_low_signal=exclude_low_signal,
        )
        if candidate_idx.size == 0:
            raise ValueError("no candidate tracks available")

        za_cand = self.za_all[torch.from_numpy(candidate_idx).to(self.device)]
        relevance_raw = (-squared_euclidean_cost(za_seed, za_cand).squeeze(0)).detach().cpu().numpy().astype(np.float32)
        relevance_all = _minmax(relevance_raw)

        if mode == "open":
            recall_n = min(max(k, recall_k), int(candidate_idx.shape[0]))
            recall_local = np.argsort(-relevance_all)[:recall_n]
            candidate_idx = candidate_idx[recall_local]
            relevance = relevance_all[recall_local]
            relevance_raw = relevance_raw[recall_local]
            za_cand = za_cand[torch.from_numpy(recall_local).to(self.device)]
        else:
            relevance = relevance_all

        zs_cand = self.zs_all[torch.from_numpy(candidate_idx).to(self.device)]
        novelty = _minmax(torch.cdist(zs_cand, zs_seed).mean(dim=1).detach().cpu().numpy())
        target_affinity = _minmax(self._target_affinity(zs_cand=zs_cand, target_culture=effective_target))
        minority = self._minority_scores(candidate_idx)
        source = self._source_scores(candidate_idx)

        final_scores = (
            float(weights.relevance) * relevance
            + float(weights.novelty) * novelty
            + float(weights.target_affinity) * target_affinity
            + float(weights.minority) * minority
            + float(weights.source) * source
        ).astype(np.float32)

        selected_local = self._select_diverse(
            candidate_idx=candidate_idx,
            zs_cand=zs_cand,
            base_scores=final_scores,
            seed_idx=np.array([], dtype=np.int64),
            k=k,
            diversity_lambda=float(weights.diversity_lambda),
            exclude_same_artist=exclude_same_artist,
        )

        recommendations: list[dict[str, Any]] = []
        for rank, local_idx in enumerate(selected_local, start=1):
            idx = int(candidate_idx[int(local_idx)])
            recommendations.append(
                self._track_payload(
                    idx,
                    rank=rank,
                    score=float(final_scores[int(local_idx)]),
                    score_components={
                        "relevance": float(relevance[int(local_idx)]),
                        "novelty": float(novelty[int(local_idx)]),
                        "target_affinity": float(target_affinity[int(local_idx)]),
                        "minority": float(minority[int(local_idx)]),
                        "source": float(source[int(local_idx)]),
                        "ot_relevance_raw": float(relevance_raw[int(local_idx)]),
                    },
                )
            )

        seed_payload = {
            "track_id": _clean(upload_info.get("track_id")) or "uploaded_audio",
            "rank": None,
            "title": _clean(upload_info.get("title")) or _clean(upload_info.get("filename")) or "Uploaded audio",
            "artist": _clean(upload_info.get("artist")) or "Uploaded audio",
            "album": "",
            "culture": _clean(seed_culture) or str(inferred["culture"]),
            "source_dataset": "upload",
            "label": "uploaded_audio",
            "country": "",
            "duration_ms": 0.0,
            "audio_is_preview": "false",
            "preview_available": "false",
            "cover_art_url": "",
            "cover_art_url_large": "",
            "platform": "upload",
            "platform_track_url": "",
            "platform_album_url": "",
            "full_track_url": "",
            "preview_url": "",
            "license_url": "",
            "audio_api_url": _clean(upload_info.get("audio_api_url")),
        }

        return {
            "ok": True,
            "algorithm": {
                "name": "dcas_mainline_uploaded_audio_recommender",
                "mode": mode,
                "backbone": "ntua-slp/CultureMERT-95M embeddings",
                "model": "dcas_full_v4_main_culturemert_stage3",
                "reranker": "uploaded CultureMERT seed -> DCAS latent encoding -> OT relevance + calibrated cultural reranking",
                "weights": {
                    "relevance": float(weights.relevance),
                    "novelty": float(weights.novelty),
                    "target_affinity": float(weights.target_affinity),
                    "minority": float(weights.minority),
                    "source": float(weights.source),
                    "diversity_lambda": float(weights.diversity_lambda),
                },
                "minority_signal": "catalog culture inverse-frequency proxy until real user logs are connected",
            },
            "request": {
                "seed_track_ids": [],
                "seed_culture": seed_culture,
                "target_culture": target_culture,
                "effective_target_culture": effective_target,
                "mode": mode,
                "k": k,
                "recall_k": recall_k,
                "exclude_same_artist": bool(exclude_same_artist),
                "exclude_low_signal": bool(exclude_low_signal),
                "random_seed": random_seed,
            },
            "upload": dict(upload_info),
            "culture_inference": inferred,
            "seeds": [seed_payload],
            "recommendations": recommendations,
            "metrics": self._result_metrics(recommendations=recommendations),
            "warnings": self._warnings(mode=mode)
            + ["uploaded audio culture is inferred from the nearest DCAS culture centroid unless seed_culture is provided."],
        }

    def audio_file(self, track_id: str) -> tuple[Path, str]:
        idx = self._require_track(track_id)
        row = self.metadata_by_id.get(str(self.tracks.track_id[idx]), {})
        audio_path = _clean(row.get("audio_path"))
        if not audio_path:
            raise FileNotFoundError(f"track has no audio_path: {track_id}")
        path = _metadata_audio_path(audio_path, self.metadata_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        return path, _media_type(path)

    def recommend(
        self,
        *,
        seed_track_ids: list[str] | None = None,
        seed_culture: str | None = None,
        target_culture: str | None = None,
        mode: str = "open",
        k: int = 10,
        recall_k: int = 600,
        random_seed: int | None = 42,
        exclude_same_artist: bool = False,
        exclude_low_signal: bool = True,
        weights: MainlineWeights | None = None,
    ) -> dict[str, Any]:
        weights = weights or MainlineWeights()
        mode = str(mode or "open").strip().lower()
        if mode not in {"open", "target"}:
            raise ValueError("mode must be one of: open, target")
        k = max(1, min(100, int(k)))
        recall_k = max(k, min(5000, int(recall_k)))

        seed_idx = self._resolve_seed_indices(
            seed_track_ids=seed_track_ids or [],
            seed_culture=seed_culture,
            random_seed=random_seed,
            exclude_low_signal=exclude_low_signal,
        )
        seed_track_ids = [str(self.tracks.track_id[i]) for i in seed_idx.tolist()]
        seed_cultures = [str(self.tracks.culture[i]) for i in seed_idx.tolist()]
        effective_target = _clean(target_culture) or (seed_cultures[0] if seed_cultures else "")
        if not effective_target:
            raise ValueError("target_culture could not be inferred")
        if effective_target not in self.culture_counts:
            raise ValueError(f"unknown target_culture={effective_target}")

        candidate_idx = self._candidate_indices(
            seed_idx=seed_idx,
            mode=mode,
            target_culture=effective_target,
            exclude_low_signal=exclude_low_signal,
        )
        if candidate_idx.size == 0:
            raise ValueError("no candidate tracks available")

        relevance_raw = self._ot_relevance(seed_idx=seed_idx, candidate_idx=candidate_idx)
        relevance_all = _minmax(relevance_raw)

        if mode == "open":
            recall_n = min(max(k, recall_k), int(candidate_idx.shape[0]))
            recall_local = np.argsort(-relevance_all)[:recall_n]
            candidate_idx = candidate_idx[recall_local]
            relevance = relevance_all[recall_local]
            relevance_raw = relevance_raw[recall_local]
        else:
            relevance = relevance_all

        zs_seed = self.zs_all[torch.from_numpy(seed_idx).to(self.device)]
        zs_cand = self.zs_all[torch.from_numpy(candidate_idx).to(self.device)]
        novelty = _minmax(torch.cdist(zs_cand, zs_seed).mean(dim=1).detach().cpu().numpy())
        target_affinity = _minmax(self._target_affinity(zs_cand=zs_cand, target_culture=effective_target))
        minority = self._minority_scores(candidate_idx)
        source = self._source_scores(candidate_idx)

        final_scores = (
            float(weights.relevance) * relevance
            + float(weights.novelty) * novelty
            + float(weights.target_affinity) * target_affinity
            + float(weights.minority) * minority
            + float(weights.source) * source
        ).astype(np.float32)

        selected_local = self._select_diverse(
            candidate_idx=candidate_idx,
            zs_cand=zs_cand,
            base_scores=final_scores,
            seed_idx=seed_idx,
            k=k,
            diversity_lambda=float(weights.diversity_lambda),
            exclude_same_artist=exclude_same_artist,
        )

        recommendations: list[dict[str, Any]] = []
        for rank, local_idx in enumerate(selected_local, start=1):
            idx = int(candidate_idx[int(local_idx)])
            recommendations.append(
                self._track_payload(
                    idx,
                    rank=rank,
                    score=float(final_scores[int(local_idx)]),
                    score_components={
                        "relevance": float(relevance[int(local_idx)]),
                        "novelty": float(novelty[int(local_idx)]),
                        "target_affinity": float(target_affinity[int(local_idx)]),
                        "minority": float(minority[int(local_idx)]),
                        "source": float(source[int(local_idx)]),
                        "ot_relevance_raw": float(relevance_raw[int(local_idx)]),
                    },
                )
            )

        return {
            "ok": True,
            "algorithm": {
                "name": "dcas_mainline_seed_recommender",
                "mode": mode,
                "backbone": "ntua-slp/CultureMERT-95M embeddings",
                "model": "dcas_full_v4_main_culturemert_stage3",
                "reranker": "OT relevance + calibrated cultural reranking",
                "weights": {
                    "relevance": float(weights.relevance),
                    "novelty": float(weights.novelty),
                    "target_affinity": float(weights.target_affinity),
                    "minority": float(weights.minority),
                    "source": float(weights.source),
                    "diversity_lambda": float(weights.diversity_lambda),
                },
                "minority_signal": "catalog culture inverse-frequency proxy until real user logs are connected",
            },
            "request": {
                "seed_track_ids": seed_track_ids,
                "seed_culture": seed_culture,
                "target_culture": target_culture,
                "effective_target_culture": effective_target,
                "mode": mode,
                "k": k,
                "recall_k": recall_k,
                "exclude_same_artist": bool(exclude_same_artist),
                "exclude_low_signal": bool(exclude_low_signal),
                "random_seed": random_seed,
            },
            "seeds": [self._track_payload(int(i)) for i in seed_idx.tolist()],
            "recommendations": recommendations,
            "metrics": self._result_metrics(recommendations=recommendations),
            "warnings": self._warnings(mode=mode),
        }

    def _load_metadata(self, path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
        rows: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            for row in reader:
                tid = _clean(row.get("track_id"))
                if tid and tid not in rows:
                    rows[tid] = {str(k): _clean(v) for k, v in row.items()}
        return rows, fields

    def _encode_external_embedding(self, embedding: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if int(emb.shape[0]) != int(self.tracks.dim):
            raise ValueError(f"external embedding dim={emb.shape[0]} does not match catalog dim={self.tracks.dim}")
        if not np.isfinite(emb).all():
            raise ValueError("external embedding contains non-finite values")
        with torch.no_grad():
            x = torch.from_numpy(emb[None, :].astype(np.float32)).to(self.device)
            _, zs_mu, za_mu = self.model.encode(x)
        return emb, zs_mu.detach(), za_mu.detach()

    def _infer_culture(self, zs_seed: torch.Tensor) -> dict[str, Any]:
        d = torch.cdist(zs_seed, self.culture_centroids)
        probs = torch.softmax(-d, dim=1).detach().cpu().numpy()[0]
        order = np.argsort(-probs).tolist()
        candidates = [
            {
                "culture": str(self.culture_names[int(i)]),
                "probability": float(probs[int(i)]),
            }
            for i in order[:5]
        ]
        best = candidates[0] if candidates else {"culture": "", "probability": 0.0}
        return {
            "culture": str(best["culture"]),
            "confidence": float(best["probability"]),
            "top_candidates": candidates,
        }

    def _encode_catalog(self, batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
        if int(self.model.cfg.in_dim) != int(self.tracks.dim):
            raise ValueError(f"model in_dim={self.model.cfg.in_dim} does not match tracks dim={self.tracks.dim}")
        zs_blocks: list[torch.Tensor] = []
        za_blocks: list[torch.Tensor] = []
        with torch.no_grad():
            emb = self.tracks.embedding.astype(np.float32)
            for start in range(0, int(emb.shape[0]), int(batch_size)):
                batch = torch.from_numpy(emb[start : start + int(batch_size)]).to(self.device)
                _, zs_mu, za_mu = self.model.encode(batch)
                zs_blocks.append(zs_mu.detach())
                za_blocks.append(za_mu.detach())
        return torch.cat(zs_blocks, dim=0), torch.cat(za_blocks, dim=0)

    def _build_culture_centroids(self) -> tuple[list[str], torch.Tensor]:
        names = sorted(self.culture_counts.keys())
        centroids: list[torch.Tensor] = []
        culture_arr = self.tracks.culture.astype(str)
        for name in names:
            idx = np.nonzero(culture_arr == str(name))[0]
            if idx.size == 0:
                continue
            centroids.append(self.zs_all[torch.from_numpy(idx).to(self.device)].mean(dim=0, keepdim=True))
        if not centroids:
            raise ValueError("no culture centroids available")
        return names, torch.cat(centroids, dim=0)

    def _require_track(self, track_id: str) -> int:
        tid = _clean(track_id)
        if tid not in self.track_id_to_idx:
            raise KeyError(f"track_id not found: {track_id}")
        return int(self.track_id_to_idx[tid])

    def _resolve_seed_indices(
        self,
        *,
        seed_track_ids: list[str],
        seed_culture: str | None,
        random_seed: int | None,
        exclude_low_signal: bool,
    ) -> np.ndarray:
        idx: list[int] = []
        for tid in seed_track_ids:
            if _clean(tid):
                idx.append(self._require_track(tid))
        if idx:
            return np.array(sorted(set(idx)), dtype=np.int64)

        all_idx = np.arange(len(self.tracks), dtype=np.int64)
        if _clean(seed_culture):
            culture = _clean(seed_culture)
            all_idx = all_idx[self.tracks.culture.astype(str)[all_idx] == culture]
        if exclude_low_signal:
            all_idx = np.array([int(i) for i in all_idx.tolist() if not self._is_low_signal(int(i))], dtype=np.int64)
        if all_idx.size == 0:
            raise ValueError("no seed candidates available")
        rng = np.random.default_rng(42 if random_seed is None else int(random_seed))
        return np.array([int(rng.choice(all_idx))], dtype=np.int64)

    def _candidate_indices(
        self,
        *,
        seed_idx: np.ndarray,
        mode: str,
        target_culture: str,
        exclude_low_signal: bool,
    ) -> np.ndarray:
        seed_set = {int(i) for i in seed_idx.tolist()}
        if mode == "target":
            base = np.nonzero(self.tracks.culture.astype(str) == str(target_culture))[0].astype(np.int64)
        else:
            base = np.arange(len(self.tracks), dtype=np.int64)
        keep: list[int] = []
        for idx in base.tolist():
            if int(idx) in seed_set:
                continue
            if exclude_low_signal and self._is_low_signal(int(idx)):
                continue
            keep.append(int(idx))
        return np.array(keep, dtype=np.int64)

    def _ot_relevance(self, *, seed_idx: np.ndarray, candidate_idx: np.ndarray) -> np.ndarray:
        za_seed = self.za_all[torch.from_numpy(seed_idx).to(self.device)]
        za_cand = self.za_all[torch.from_numpy(candidate_idx).to(self.device)]
        cost = squared_euclidean_cost(za_seed, za_cand)
        if int(seed_idx.shape[0]) == 1:
            return (-cost.squeeze(0)).detach().cpu().numpy().astype(np.float32)
        a = torch.full((int(seed_idx.shape[0]),), 1.0 / int(seed_idx.shape[0]), device=self.device)
        b = torch.full((int(candidate_idx.shape[0]),), 1.0 / int(candidate_idx.shape[0]), device=self.device)
        plan = sinkhorn_plan(a=a, b=b, cost=cost, epsilon=0.1, iters=200)
        col_mass = plan.sum(dim=0).clamp_min(1e-12)
        col_avg_cost = (plan * cost).sum(dim=0) / col_mass
        return (-col_avg_cost).detach().cpu().numpy().astype(np.float32)

    def _target_affinity(self, *, zs_cand: torch.Tensor, target_culture: str) -> np.ndarray:
        d = torch.cdist(zs_cand, self.culture_centroids)
        probs = torch.softmax(-d, dim=1)
        if str(target_culture) not in self.culture_names:
            return np.zeros((int(zs_cand.shape[0]),), dtype=np.float32)
        col = self.culture_names.index(str(target_culture))
        return probs[:, col].detach().cpu().numpy().astype(np.float32)

    def _minority_scores(self, candidate_idx: np.ndarray) -> np.ndarray:
        counts = self.culture_counts
        raw = np.array([1.0 / max(1, counts.get(str(self.tracks.culture[int(i)]), 1)) for i in candidate_idx.tolist()], dtype=np.float32)
        return _minmax(raw)

    def _source_scores(self, candidate_idx: np.ndarray) -> np.ndarray:
        if self.tracks.source_dataset is None:
            return np.zeros((int(candidate_idx.shape[0]),), dtype=np.float32)
        raw = np.array(
            [1.0 / max(1, self.source_counts.get(str(self.tracks.source_dataset[int(i)]), 1)) for i in candidate_idx.tolist()],
            dtype=np.float32,
        )
        return _minmax(raw)

    def _select_diverse(
        self,
        *,
        candidate_idx: np.ndarray,
        zs_cand: torch.Tensor,
        base_scores: np.ndarray,
        seed_idx: np.ndarray,
        k: int,
        diversity_lambda: float,
        exclude_same_artist: bool,
    ) -> list[int]:
        zs = zs_cand.detach().cpu().numpy().astype(np.float64)
        zs = zs / np.maximum(1e-12, np.linalg.norm(zs, axis=1, keepdims=True))
        seed_keys = {self._title_artist_key(int(i)) for i in seed_idx.tolist()}
        seed_artists = {self._artist_key(int(i)) for i in seed_idx.tolist() if self._artist_key(int(i))}
        used_keys = set(seed_keys)
        selected: list[int] = []
        remaining: set[int] = set(range(int(candidate_idx.shape[0])))
        while remaining and len(selected) < int(k):
            best_idx: int | None = None
            best_score: float | None = None
            for local_idx in remaining:
                idx = int(candidate_idx[int(local_idx)])
                key = self._title_artist_key(idx)
                artist = self._artist_key(idx)
                if key and key in used_keys:
                    continue
                if exclude_same_artist and artist and artist in seed_artists:
                    continue
                penalty = 0.0
                if selected:
                    penalty = float(np.max(zs[int(local_idx)] @ zs[np.array(selected, dtype=np.int64)].T))
                score = float(base_scores[int(local_idx)]) - float(diversity_lambda) * penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = int(local_idx)
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            selected_key = self._title_artist_key(int(candidate_idx[best_idx]))
            if selected_key:
                used_keys.add(selected_key)
        if len(selected) < int(k):
            for local_idx in np.argsort(-base_scores).tolist():
                if int(local_idx) in selected:
                    continue
                idx = int(candidate_idx[int(local_idx)])
                key = self._title_artist_key(idx)
                if key and key in used_keys:
                    continue
                selected.append(int(local_idx))
                if key:
                    used_keys.add(key)
                if len(selected) >= int(k):
                    break
        return selected[: int(k)]

    def _track_payload(
        self,
        idx: int,
        *,
        rank: int | None = None,
        score: float | None = None,
        score_components: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        track_id = str(self.tracks.track_id[int(idx)])
        row = self.metadata_by_id.get(track_id, {})
        payload: dict[str, Any] = {
            "track_id": track_id,
            "rank": rank,
            "title": _clean(row.get("title")) or track_id,
            "artist": _clean(row.get("artist")) or _clean(row.get("source_dataset")),
            "album": _clean(row.get("album")),
            "culture": str(self.tracks.culture[int(idx)]),
            "source_dataset": str(self.tracks.source_dataset[int(idx)]) if self.tracks.source_dataset is not None else _clean(row.get("source_dataset")),
            "label": _clean(row.get("label")),
            "country": _clean(row.get("country")),
            "duration_ms": _safe_float(row.get("duration_ms"), default=0.0),
            "audio_is_preview": _clean(row.get("audio_is_preview")),
            "preview_available": _clean(row.get("preview_available")),
            "cover_art_url": _clean(row.get("cover_art_url")) or _clean(row.get("artwork_url_large")) or _clean(row.get("image_url")),
            "cover_art_url_large": _clean(row.get("cover_art_url_large")) or _clean(row.get("cover_art_url")) or _clean(row.get("artwork_url_large")),
            "platform": _clean(row.get("platform")) or _clean(row.get("source_dataset")),
            "platform_track_url": _clean(row.get("platform_track_url")) or _clean(row.get("track_url")) or _clean(row.get("jamendo_url")),
            "platform_album_url": _clean(row.get("platform_album_url")) or _clean(row.get("collection_url")),
            "full_track_url": _clean(row.get("full_track_url")) or _clean(row.get("jamendo_url")),
            "preview_url": _clean(row.get("preview_url")) or _clean(row.get("audio_url")),
            "license_url": _clean(row.get("license_url")),
            "audio_api_url": f"/api/mainline/audio/{track_id}",
        }
        if score is not None:
            payload["score"] = float(score)
        if score_components is not None:
            payload["score_components"] = {k: float(v) for k, v in score_components.items()}
        return payload

    def _result_metrics(self, *, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        cultures = Counter(_clean(item.get("culture")) for item in recommendations)
        sources = Counter(_clean(item.get("source_dataset")) for item in recommendations)
        scores = [_safe_float(item.get("score"), default=float("nan")) for item in recommendations if item.get("score") is not None]
        return {
            "n": int(len(recommendations)),
            "culture_counts": dict(sorted(cultures.items())),
            "source_counts": dict(sorted(sources.items())),
            "mean_score": float(np.mean(scores)) if scores else None,
            "with_cover_art": int(sum(1 for item in recommendations if _clean(item.get("cover_art_url")))),
            "with_platform_link": int(sum(1 for item in recommendations if _clean(item.get("platform_track_url")))),
        }

    def _warnings(self, *, mode: str) -> list[str]:
        warnings: list[str] = []
        warnings.append("30k catalog does not yet have real user interaction logs; minority uses a catalog-balance proxy.")
        if mode == "open":
            warnings.append("open mode is the product seed-track adaptation of the mainline; strict benchmark mode is target.")
        return warnings

    def _is_low_signal(self, idx: int) -> bool:
        row = self.metadata_by_id.get(str(self.tracks.track_id[int(idx)]), {})
        text = " ".join(
            [
                _norm_key(row.get("title")),
                _norm_key(row.get("album")),
                _norm_key(row.get("label")),
                _norm_key(row.get("tags")),
            ]
        )
        return any(term in text for term in LOW_SIGNAL_TERMS)

    def _matches_query(self, idx: int, query_terms: list[str]) -> bool:
        row = self.metadata_by_id.get(str(self.tracks.track_id[int(idx)]), {})
        text = " ".join(
            [
                _norm_key(self.tracks.track_id[int(idx)]),
                _norm_key(self.tracks.culture[int(idx)]),
                _norm_key(self.tracks.source_dataset[int(idx)] if self.tracks.source_dataset is not None else ""),
                _norm_key(row.get("title")),
                _norm_key(row.get("artist")),
                _norm_key(row.get("album")),
                _norm_key(row.get("label")),
                _norm_key(row.get("country")),
                _norm_key(row.get("tags")),
            ]
        )
        return all(term in text for term in query_terms)

    def _title_artist_key(self, idx: int) -> str:
        row = self.metadata_by_id.get(str(self.tracks.track_id[int(idx)]), {})
        title = _norm_key(row.get("title"))
        artist = _norm_key(row.get("artist"))
        return f"{title}|{artist}" if title or artist else ""

    def _artist_key(self, idx: int) -> str:
        row = self.metadata_by_id.get(str(self.tracks.track_id[int(idx)]), {})
        return _norm_key(row.get("artist"))


_PLATFORMS: dict[tuple[str, bool], MainlineRecommendationPlatform] = {}


def get_mainline_platform(storage: Storage, *, prefer_cuda: bool = False) -> MainlineRecommendationPlatform:
    key = (str(storage.root.resolve()), bool(prefer_cuda and torch.cuda.is_available()))
    platform = _PLATFORMS.get(key)
    if platform is None:
        platform = MainlineRecommendationPlatform(storage=storage, prefer_cuda=bool(prefer_cuda))
        _PLATFORMS[key] = platform
    return platform
