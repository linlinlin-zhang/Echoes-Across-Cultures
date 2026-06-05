from __future__ import annotations

import base64
import ast
import csv
import hashlib
import json
import math
import mimetypes
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .paths import Storage
from .schemas import (
    PrototypeAnalyzeRequest,
    PrototypeFeedbackRequest,
    PrototypeRegisterRequest,
)


def create_prototype_router(storage: Storage) -> APIRouter:
    service = PrototypeService(storage)
    router = APIRouter(prefix="/api/prototype", tags=["prototype"])

    @router.get("/bootstrap")
    def bootstrap() -> dict[str, Any]:
        return service.bootstrap_payload()

    @router.post("/register")
    def register(req: PrototypeRegisterRequest) -> dict[str, Any]:
        return service.register_profile(req.name, req.email, req.role)

    @router.post("/upload")
    async def upload_audio(file: UploadFile = File(...)) -> dict[str, Any]:
        return await service.save_upload(file)

    @router.post("/analyze")
    def analyze(req: PrototypeAnalyzeRequest) -> dict[str, Any]:
        return service.analyze_upload(
            upload_id=req.upload_id,
            mode=req.mode,
            lens=req.lens,
            engine=req.engine,
            top_k=req.top_k,
        )

    @router.post("/feedback")
    def save_feedback(req: PrototypeFeedbackRequest) -> dict[str, Any]:
        return service.record_feedback(
            track=req.track,
            recommendation_id=req.recommendation_id,
            rating=req.rating,
            comment=req.comment,
            profile_id=req.profile_id,
        )

    @router.get("/uploads/{upload_id}/audio")
    def stream_upload(upload_id: str) -> FileResponse:
        return serve_audio_file(service.get_upload_audio_path(upload_id))

    @router.get("/catalog/{track_id}/audio")
    def stream_catalog(track_id: str) -> FileResponse:
        return serve_audio_file(service.get_catalog_audio_path(track_id))

    return router


def serve_audio_file(file_path: Path) -> FileResponse:
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="audio file not found")
    media_type, _ = mimetypes.guess_type(file_path.name)
    return FileResponse(
        str(file_path),
        media_type=media_type or "audio/mpeg",
        headers={"Accept-Ranges": "bytes"},
    )


class PrototypeService:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.prototype_dir = storage.ensure_dir("prototype")
        self.upload_dir = storage.ensure_dir("prototype/uploads")
        self.db_path = storage.resolve_rel("prototype/prototype.sqlite3")
        self.catalog_path, self.embedding_path, self.catalog_source = resolve_catalog_sources(storage)
        self.catalog_vector_dim = 768
        self._embedding_lookup = load_embedding_lookup(self.embedding_path)
        if self._embedding_lookup:
            first_vector = next(iter(self._embedding_lookup.values()))
            if first_vector:
                self.catalog_vector_dim = len(first_vector)
        self._ensure_db()
        self.catalog = self._load_catalog(limit=84)

    def bootstrap_payload(self) -> dict[str, Any]:
        sample_track = next((item for item in self.catalog if item.get("audio_path")), None)
        return {
            "profile": self._get_latest_profile(),
            "feedback": self._recent_feedback(limit=8),
            "stats": self._stats(),
            "catalog_size": len(self.catalog),
            "catalog_source": self.catalog_source,
            "catalog_vector_dim": self.catalog_vector_dim,
            "catalog_origins": list(dict.fromkeys(item["origin"] for item in self.catalog))[:10],
            "provider": self._provider_mode(),
            "llm_provider": self._llm_provider_mode(),
            "sample_track": {
                "id": sample_track["id"],
                "title": sample_track["title"],
                "origin": sample_track["origin"],
                "descriptor": sample_track["descriptor"],
                "audio_url": f"/api/prototype/catalog/{sample_track['id']}/audio",
            }
            if sample_track
            else None,
        }

    def register_profile(self, name: str, email: str, role: str) -> dict[str, Any]:
        now = time.time()
        profile_id = f"profile_{uuid.uuid4().hex[:10]}"
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM listener_profiles WHERE email = ?",
                (email,),
            ).fetchone()
            if existing:
                profile_id = existing["id"]
                conn.execute(
                    """
                    UPDATE listener_profiles
                    SET name = ?, role = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (name, role, now, profile_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO listener_profiles (id, name, email, role, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (profile_id, name, email, role, now, now),
                )
        return {
            "profile": {
                "id": profile_id,
                "name": name,
                "email": email,
                "role": role,
            },
            "message": "听众档案已保存，之后的推荐和反馈会更容易与你的偏好关联。",
        }

    async def save_upload(self, file: UploadFile) -> dict[str, Any]:
        filename = Path(file.filename or "audio.bin").name
        suffix = Path(filename).suffix.lower() or ".bin"
        upload_id = f"upload_{uuid.uuid4().hex[:12]}"
        relative_path = f"prototype/uploads/{upload_id}{suffix}"
        target_path = self.storage.resolve_rel(relative_path)

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty file")
        target_path.write_bytes(content)

        descriptor = guess_descriptor(filename)
        waveform = build_waveform_points(content, bins=64)
        checksum = hashlib.sha1(content).hexdigest()
        now = time.time()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO uploads (
                    id, original_name, stored_relpath, mime_type, size_bytes, descriptor,
                    waveform_json, checksum_sha1, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    filename,
                    relative_path,
                    file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
                    len(content),
                    descriptor,
                    json.dumps(waveform, ensure_ascii=False),
                    checksum,
                    now,
                ),
            )

        return {
            "upload": {
                "id": upload_id,
                "name": filename,
                "size_mb": round(len(content) / (1024 * 1024), 2),
                "descriptor": descriptor,
                "waveform": waveform,
                "audio_url": f"/api/prototype/uploads/{upload_id}/audio",
            },
            "message": "音频已上传，可以开始生成嵌入与推荐结果。",
        }

    def analyze_upload(self, upload_id: str, mode: str, lens: str, engine: str, top_k: int) -> dict[str, Any]:
        upload = self._get_upload(upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="upload not found")

        audio_path = self.storage.resolve_rel(upload["stored_relpath"])
        audio_bytes = audio_path.read_bytes()
        embedding_result = self._build_embedding(audio_bytes, upload["original_name"], upload["descriptor"])
        base_recommendations = self._recommend(
            source_name=upload["original_name"],
            source_descriptor=upload["descriptor"],
            source_embedding=embedding_result["vector"],
            mode=mode,
            lens=lens,
            top_k=max(top_k, 8),
        )
        recommendations, engine_warning = self._rerank_recommendations(
            engine=engine,
            upload_name=upload["original_name"],
            upload_descriptor=upload["descriptor"],
            mode=mode,
            lens=lens,
            recommendations=base_recommendations,
            top_k=top_k,
        )
        top_recommendation = recommendations[0]
        analysis_id = f"analysis_{uuid.uuid4().hex[:10]}"
        source_bpm = 84 + int(seeded_value(hash_string(upload["original_name"])) * 24)
        stages = build_stage_log(upload["original_name"], mode, lens)
        now = time.time()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    id, upload_id, mode, lens, embedding_dim, source_bpm,
                    bridge_score, provider_mode, recommendations_json, stages_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    upload_id,
                    mode,
                    lens,
                    embedding_result["reported_dim"],
                    source_bpm,
                    top_recommendation["bridge"],
                    embedding_result["provider"],
                    json.dumps(recommendations, ensure_ascii=False),
                    json.dumps(stages, ensure_ascii=False),
                    now,
                ),
            )

        return {
            "analysis": {
                "id": analysis_id,
                "mode": mode,
                "lens": lens,
                "engine": engine,
                "embedding_dim": embedding_result["reported_dim"],
                "bridge_score": top_recommendation["bridge"],
                "source_bpm": source_bpm,
                "provider": embedding_result["provider"],
                "provider_warning": embedding_result.get("warning"),
                "engine_warning": engine_warning,
            },
            "upload": {
                "id": upload["id"],
                "name": upload["original_name"],
                "size_mb": round(upload["size_bytes"] / (1024 * 1024), 2),
                "descriptor": upload["descriptor"],
                "waveform": json.loads(upload["waveform_json"]),
                "audio_url": f"/api/prototype/uploads/{upload_id}/audio",
            },
            "stages": stages,
            "recommendations": recommendations,
            "stats": self._stats(),
        }

    def record_feedback(
        self,
        track: str,
        recommendation_id: str | None,
        rating: int,
        comment: str,
        profile_id: str | None,
    ) -> dict[str, Any]:
        now = time.time()
        feedback_id = f"feedback_{uuid.uuid4().hex[:10]}"
        profile = self._get_profile(profile_id) if profile_id else None
        profile_name = profile["name"] if profile else "匿名听众"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    id, profile_id, profile_name, track, recommendation_id,
                    rating, comment, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    profile_id,
                    profile_name,
                    track,
                    recommendation_id,
                    rating,
                    comment.strip(),
                    now,
                ),
            )

        return {
            "feedback": {
                "id": feedback_id,
                "track": track,
                "rating": rating,
                "comment": comment.strip(),
                "profile_name": profile_name,
            },
            "recent_feedback": self._recent_feedback(limit=8),
            "stats": self._stats(),
            "message": "反馈已保存，之后的推荐会越来越懂你的口味。",
        }

    def get_upload_audio_path(self, upload_id: str) -> Path:
        upload = self._get_upload(upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="upload not found")
        return self.storage.resolve_rel(upload["stored_relpath"])

    def get_catalog_audio_path(self, track_id: str) -> Path:
        item = next((item for item in self.catalog if item["id"] == track_id), None)
        if item is None or not item["audio_path"]:
            raise HTTPException(status_code=404, detail="catalog track not found")
        path = Path(item["audio_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="catalog audio missing")
        return path

    def _load_catalog(self, limit: int) -> list[dict[str, Any]]:
        if not self.catalog_path or not self.catalog_path.exists() or not self._embedding_lookup:
            return build_fallback_catalog(self.catalog_vector_dim)

        selected: list[dict[str, Any]] = []
        per_culture: dict[str, int] = {}
        with self.catalog_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                track_id = (row.get("track_id") or "").strip()
                audio_path = (row.get("audio_path") or "").strip()
                culture = (row.get("culture") or "").strip()
                if not track_id or not audio_path or not culture:
                    continue
                if row.get("dedup_keep") not in {"1", "", None}:
                    continue
                if per_culture.get(culture, 0) >= 12:
                    continue

                path_obj = Path(audio_path)
                if not path_obj.exists():
                    continue
                vector = self._embedding_lookup.get(track_id)
                if not vector:
                    continue

                descriptor = build_catalog_descriptor(row)
                item = {
                    "id": track_id,
                    "title": build_catalog_title(row),
                    "origin": format_origin_label(
                        culture,
                        row.get("region") or row.get("source_dataset") or culture,
                    ),
                    "culture": culture,
                    "artist": humanize_title(
                        first_metadata_term(row, "artist")
                        or first_metadata_term(row, "source_dataset")
                        or format_origin_label(culture, culture)
                    ),
                    "audio_path": audio_path,
                    "tags": build_catalog_tags(row),
                    "descriptor": descriptor,
                    "vector": vector,
                    "bpm": 72 + int(seeded_value(hash_string(track_id)) * 54),
                }
                selected.append(item)
                per_culture[culture] = per_culture.get(culture, 0) + 1
                if len(selected) >= limit:
                    break

        return selected or build_fallback_catalog(self.catalog_vector_dim)

    def _recommend(
        self,
        source_name: str,
        source_descriptor: str,
        source_embedding: list[float],
        mode: str,
        lens: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        weights = {
            "bridge": {"bridge": 1.12, "novelty": 0.82, "similarity": 1.0},
            "novelty": {"bridge": 0.92, "novelty": 1.24, "similarity": 0.8},
            "precision": {"bridge": 0.86, "novelty": 0.72, "similarity": 1.22},
        }.get(mode, {"bridge": 1.0, "novelty": 1.0, "similarity": 1.0})
        source_seed = hash_string(f"{source_name}|{source_descriptor}|{mode}|{lens}")

        scored: list[dict[str, Any]] = []
        for index, item in enumerate(self.catalog):
            similarity = cosine_similarity(source_embedding, item["vector"])
            novelty = clamp(0.48 + abs(0.75 - similarity) * 0.85)
            bridge = clamp((similarity * 0.62) + (0.32 * lens_bonus(item, lens)) + 0.12)
            confidence = clamp((similarity * 0.55) + (bridge * 0.35) + 0.08)
            variation = seeded_value(source_seed + index * 37)
            score = (
                similarity * weights["similarity"] + novelty * weights["novelty"] + bridge * weights["bridge"]
            ) * 25 + variation * 2.8

            scored.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "origin": item["origin"],
                    "bridge": round(bridge, 2),
                    "novelty": round(novelty, 2),
                    "similarity": round(similarity, 2),
                    "confidence": round(confidence, 2),
                    "score": int(round(score)),
                    "bpm": str(item["bpm"]),
                    "axis": choose_axis(item, lens),
                    "summary": build_summary(item, similarity, bridge, novelty),
                    "reason": build_reason(item, lens, similarity, bridge),
                    "tags": item["tags"][:4],
                    "audio_url": f"/api/prototype/catalog/{item['id']}/audio" if item["audio_path"] else None,
                }
            )

        scored.sort(key=lambda value: value["score"], reverse=True)
        distinct_origins: set[str] = set()
        picked: list[dict[str, Any]] = []
        for item in scored:
            if item["origin"] in distinct_origins and len(picked) < top_k:
                continue
            picked.append(item)
            distinct_origins.add(item["origin"])
            if len(picked) >= top_k:
                break
        return picked if picked else scored[:top_k]

    def _rerank_recommendations(
        self,
        engine: str,
        upload_name: str,
        upload_descriptor: str,
        mode: str,
        lens: str,
        recommendations: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if engine != "llm":
            return recommendations[:top_k], None

        provider_url = os.environ.get("ECHO_LLM_RECOMMENDER_URL")
        api_key = os.environ.get("ECHO_LLM_RECOMMENDER_API_KEY")
        if not provider_url:
            return recommendations[:top_k], "外部大模型推荐接口尚未配置，已自动回退到 Echo 引擎。"

        try:
            reranked = request_external_llm_recommendations(
                provider_url=provider_url,
                api_key=api_key,
                upload_name=upload_name,
                upload_descriptor=upload_descriptor,
                mode=mode,
                lens=lens,
                recommendations=recommendations[:12],
                top_k=top_k,
            )
        except RuntimeError as exc:
            return recommendations[:top_k], str(exc)
        merged = reranked[:]
        used_ids = {item["id"] for item in merged}
        for item in recommendations:
            if item["id"] in used_ids:
                continue
            merged.append(item)
            if len(merged) >= top_k:
                break
        return merged[:top_k], None

    def _build_embedding(self, audio_bytes: bytes, filename: str, descriptor: str) -> dict[str, Any]:
        provider_url = os.environ.get("ECHO_EMBEDDING_API_URL")
        api_key = os.environ.get("ECHO_EMBEDDING_API_KEY")
        if provider_url:
            try:
                result = request_external_embedding(
                    provider_url=provider_url,
                    api_key=api_key,
                    filename=filename,
                    descriptor=descriptor,
                    audio_bytes=audio_bytes,
                )
                result["vector"] = align_vector_dim(result["vector"], self.catalog_vector_dim)
                result["reported_dim"] = self.catalog_vector_dim
                return result
            except RuntimeError as exc:
                local = local_embedding(audio_bytes, filename, descriptor, self.catalog_vector_dim)
                local["warning"] = str(exc)
                return local
        return local_embedding(audio_bytes, filename, descriptor, self.catalog_vector_dim)

    def _provider_mode(self) -> str:
        return "external-configured" if os.environ.get("ECHO_EMBEDDING_API_URL") else "local-fallback"

    def _llm_provider_mode(self) -> str:
        return "external-configured" if os.environ.get("ECHO_LLM_RECOMMENDER_URL") else "local-fallback"

    def _stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            upload_count = conn.execute("SELECT COUNT(*) AS count FROM uploads").fetchone()["count"]
            analysis_count = conn.execute("SELECT COUNT(*) AS count FROM analyses").fetchone()["count"]
            feedback_rows = conn.execute("SELECT rating FROM feedback").fetchall()
            profile_count = conn.execute("SELECT COUNT(*) AS count FROM listener_profiles").fetchone()["count"]

        average_rating = None
        if feedback_rows:
            average_rating = round(
                sum(row["rating"] for row in feedback_rows) / len(feedback_rows),
                1,
            )

        return {
            "uploads": upload_count,
            "analyses": analysis_count,
            "feedback_count": len(feedback_rows),
            "profile_count": profile_count,
            "average_rating": average_rating,
        }

    def _recent_feedback(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, profile_name, track, recommendation_id, rating, comment, created_at
                FROM feedback
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "profile_name": row["profile_name"],
                "track": row["track"],
                "recommendation_id": row["recommendation_id"],
                "rating": row["rating"],
                "comment": row["comment"],
                "created_at": format_timestamp(row["created_at"]),
            }
            for row in rows
        ]

    def _get_latest_profile(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, email, role
                FROM listener_profiles
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "role": row["role"],
        }

    def _get_profile(self, profile_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, name, email, role FROM listener_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()

    def _get_upload(self, upload_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT id, original_name, stored_relpath, mime_type, size_bytes, descriptor, waveform_json
                FROM uploads
                WHERE id = ?
                """,
                (upload_id,),
            ).fetchone()

    def _ensure_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS listener_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    stored_relpath TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    descriptor TEXT NOT NULL,
                    waveform_json TEXT NOT NULL,
                    checksum_sha1 TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    upload_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    lens TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    source_bpm INTEGER NOT NULL,
                    bridge_score REAL NOT NULL,
                    provider_mode TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL,
                    stages_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(upload_id) REFERENCES uploads(id)
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT,
                    profile_name TEXT NOT NULL,
                    track TEXT NOT NULL,
                    recommendation_id TEXT,
                    rating INTEGER NOT NULL,
                    comment TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(profile_id) REFERENCES listener_profiles(id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def resolve_catalog_sources(storage: Storage) -> tuple[Path | None, Path | None, str]:
    candidates = [
        (
            "public/research_dataset_v2/metadata_v2_main_clean.csv",
            "public/research_dataset_v2/tracks_culturemert_v2_main.npz",
            "research_dataset_v2_clean",
        ),
        (
            "public/research_dataset_v2/metadata_v2_main.csv",
            "public/research_dataset_v2/tracks_culturemert_v2_main.npz",
            "research_dataset_v2",
        ),
        (
            "public/research_dataset_v1/metadata_merged.csv",
            "public/research_dataset_v1/tracks.npz",
            "research_dataset_v1",
        ),
        (
            "public/research_dataset_v4/main/metadata_release.csv",
            "public/research_dataset_v4/main/tracks_release.npz",
            "research_dataset_v4",
        ),
    ]

    for metadata_rel, embedding_rel, source_name in candidates:
        metadata_path = storage.resolve_rel(metadata_rel)
        embedding_path = storage.resolve_rel(embedding_rel)
        if metadata_path.exists() and embedding_path.exists():
            return metadata_path, embedding_path, source_name

    return None, None, "fallback-demo"


def load_embedding_lookup(npz_path: Path | None) -> dict[str, list[float]]:
    if npz_path is None or not npz_path.exists():
        return {}

    data = np.load(npz_path, allow_pickle=True)
    track_ids = data.get("track_id")
    embeddings = data.get("embedding")
    if track_ids is None or embeddings is None:
        return {}

    lookup: dict[str, list[float]] = {}
    for track_id, vector in zip(track_ids, embeddings):
        key = str(track_id).strip()
        if not key:
            continue
        lookup[key] = align_vector_dim(np.asarray(vector, dtype=float).tolist(), int(len(vector)))
    return lookup


def request_external_embedding(
    provider_url: str,
    api_key: str | None,
    filename: str,
    descriptor: str,
    audio_bytes: bytes,
) -> dict[str, Any]:
    payload = {
        "filename": filename,
        "descriptor": descriptor,
        "content_base64": base64.b64encode(audio_bytes).decode("ascii"),
    }
    request = Request(
        provider_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urlopen(request, timeout=25) as response:
            content = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"外部嵌入服务返回错误：HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接外部嵌入服务：{exc.reason}") from exc

    embedding = content.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("外部嵌入服务没有返回合法的 embedding 数组")

    return {
        "provider": content.get("provider") or "external",
        "reported_dim": int(content.get("embedding_dim") or len(embedding)),
        "vector": [float(value) for value in embedding],
    }


def request_external_llm_recommendations(
    provider_url: str,
    api_key: str | None,
    upload_name: str,
    upload_descriptor: str,
    mode: str,
    lens: str,
    recommendations: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    payload = {
        "upload_name": upload_name,
        "upload_descriptor": upload_descriptor,
        "mode": mode,
        "lens": lens,
        "top_k": top_k,
        "candidates": recommendations,
    }
    request = Request(
        provider_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urlopen(request, timeout=35) as response:
            content = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"外部大模型推荐接口返回错误：HTTP {exc.code}，已回退到 Echo 引擎。") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接外部大模型推荐接口：{exc.reason}，已回退到 Echo 引擎。") from exc

    items = content.get("recommendations")
    if not isinstance(items, list) or not items:
        raise RuntimeError("外部大模型推荐接口没有返回合法结果，已回退到 Echo 引擎。")

    by_id = {item["id"]: item for item in recommendations}
    reranked: list[dict[str, Any]] = []
    for external in items:
        item_id = external.get("id")
        if item_id not in by_id:
            continue
        merged = dict(by_id[item_id])
        if isinstance(external.get("reason"), str) and external["reason"].strip():
            merged["reason"] = external["reason"].strip()
        if isinstance(external.get("summary"), str) and external["summary"].strip():
            merged["summary"] = external["summary"].strip()
        if external.get("score") is not None:
            try:
                merged["score"] = int(external["score"])
            except (TypeError, ValueError):
                pass
        reranked.append(merged)

    if not reranked:
        raise RuntimeError("外部大模型推荐接口没有返回可匹配的候选结果，已回退到 Echo 引擎。")
    return reranked


def local_embedding(audio_bytes: bytes, filename: str, descriptor: str, dim: int) -> dict[str, Any]:
    seed = hash_string(f"{filename}|{descriptor}|{hashlib.sha1(audio_bytes).hexdigest()}")
    return {
        "provider": "local-fallback",
        "reported_dim": dim,
        "vector": deterministic_vector(f"{seed}|{descriptor}", dim),
    }


def deterministic_vector(text: str, dim: int) -> list[float]:
    values: list[float] = []
    for index in range(dim):
        token = f"{text}|{index}".encode("utf-8")
        digest = hashlib.sha1(token).digest()
        value = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        values.append((value * 2.0) - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def align_vector_dim(vector: list[float], target_dim: int) -> list[float]:
    values = [float(value) for value in vector]
    if target_dim <= 0:
        return values
    if len(values) > target_dim:
        values = values[:target_dim]
    elif len(values) < target_dim:
        values = values + ([0.0] * (target_dim - len(values)))

    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return clamp((numerator / (left_norm * right_norm) + 1.0) / 2.0)


def build_waveform_points(audio_bytes: bytes, bins: int) -> list[float]:
    if not audio_bytes:
        return [0.0] * bins
    stride = max(1, len(audio_bytes) // bins)
    waveform: list[float] = []
    for index in range(bins):
        start = index * stride
        chunk = audio_bytes[start : start + stride]
        if not chunk:
            waveform.append(0.0)
            continue
        energy = sum(abs(byte - 128) for byte in chunk[:2048]) / max(1, min(len(chunk), 2048))
        waveform.append(round(clamp(energy / 96.0), 3))
    return waveform


def build_stage_log(source_name: str, mode: str, lens: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "接收本地音轨",
            "detail": f"已接收《{source_name}》，正在整理文件信息与输入描述。",
            "progress": 14,
            "scene": "intake",
        },
        {
            "label": "生成嵌入",
            "detail": "正在请求上游嵌入流程，生成共享音乐向量表示。",
            "progress": 36,
            "scene": "embedding",
        },
        {
            "label": "投影文化因子",
            "detail": lens_note(lens),
            "progress": 58,
            "scene": "embedding",
        },
        {
            "label": "执行重排序",
            "detail": f"当前采用 {mode_label(mode)}，正在平衡桥接性、新颖度和可解释性。",
            "progress": 80,
            "scene": "recommend",
        },
        {
            "label": "拼装推荐结果",
            "detail": "推荐卡片、指标证据和推荐理由已准备完成。",
            "progress": 100,
            "scene": "recommend",
        },
    ]


def guess_descriptor(filename: str) -> str:
    lowered = filename.lower()
    if "drum" in lowered or "beat" in lowered:
        return "鼓点突出、舞蹈驱动型能量"
    if "voice" in lowered or "vocal" in lowered:
        return "人声主导、旋律叙述感较强"
    if "string" in lowered or "guitar" in lowered or "oud" in lowered:
        return "拨弦共振明显、装饰音较丰富"
    return "适合进入嵌入流程的本地音乐输入"


def parse_metadata_terms(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [humanize_title(str(item)) for item in parsed if str(item).strip()]

    return [humanize_title(raw)]


def first_metadata_term(row: dict[str, Any], key: str) -> str:
    values = parse_metadata_terms(row.get(key) or "")
    return values[0] if values else ""


def build_catalog_title(row: dict[str, Any]) -> str:
    for key in ("title", "cname", "instrument", "label"):
        value = first_metadata_term(row, key)
        if value and value != "未知":
            return value
    return humanize_title(row.get("track_id") or "Untitled")


def build_catalog_descriptor(row: dict[str, Any]) -> str:
    parts = [
        first_metadata_term(row, "title"),
        first_metadata_term(row, "cname"),
        first_metadata_term(row, "instrument"),
        first_metadata_term(row, "label"),
        first_metadata_term(row, "mood_theme"),
        first_metadata_term(row, "source_dataset"),
    ]
    clean = [part for part in parts if part and part != "未知"]
    return " / ".join(clean[:4]) or "跨文化音乐候选条目"


def build_catalog_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in (
        "title",
        "cname",
        "instrument",
        "label",
        "mood_theme",
        "language",
        "culture",
    ):
        for term in parse_metadata_terms(row.get(key) or ""):
            tags.append(translate_token(term))
    output: list[str] = []
    for tag in tags:
        if tag and tag not in output:
            output.append(tag)
    return output[:6] or ["文化候选"]


def lens_bonus(item: dict[str, Any], lens: str) -> float:
    text = f"{item['title']}|{item['descriptor']}|{' '.join(item['tags'])}".lower()
    if lens == "rhythm":
        return (
            0.78 if any(token in text for token in ["dance", "drum", "percussion", "rhythm", "pulse", "舞"]) else 0.56
        )
    if lens == "timbre":
        return 0.8 if any(token in text for token in ["voice", "string", "oud", "弦", "声", "instrument"]) else 0.58
    return 0.82 if any(token in text for token in ["ritual", "opera", "emotion", "ceremonial", "戏", "祭"]) else 0.6


def choose_axis(item: dict[str, Any], lens: str) -> str:
    descriptor = item["descriptor"]
    if lens == "rhythm":
        if "voice" in descriptor.lower() or "人声" in descriptor:
            return "声腔节奏 + 结构张力"
        return "拨弦音色 + 循环律动"
    if lens == "timbre":
        if "voice" in descriptor.lower() or "人声" in descriptor:
            return "声腔纹理 + 共鸣层次"
        return "乐器纹理 + 共振色彩"
    return "情绪轮廓 + 仪式氛围"


def build_summary(item: dict[str, Any], similarity: float, bridge: float, novelty: float) -> str:
    if bridge > 0.84:
        return f"这条候选在桥接性上最稳定，能用 {item['descriptor']} 和你的源轨建立自然联系。"
    if novelty > 0.72:
        return "这条候选的新颖度更高，适合想从熟悉音乐跳到更远文化空间的听众。"
    return "它在相似度和解释性之间比较平衡，既不会太陌生，也不会只是重复原有偏好。"


def build_reason(item: dict[str, Any], lens: str, similarity: float, bridge: float) -> str:
    return (
        f"系统选择《{item['title']}》的原因是：在 {choose_axis(item, lens)} 这条轴上，它和你的源轨形成了"
        f" {round(bridge, 2)} 的桥接强度，同时保持了 {round(similarity, 2)} 的向量邻近度。"
    )


def humanize_title(value: str) -> str:
    clean = (value or "").strip()
    if not clean:
        return "未知"
    return clean.replace("_", " ").replace("-", " ").strip()


def format_origin_label(culture: str, region: str) -> str:
    culture_map = {
        "china": "中国",
        "india": "印度",
        "turkey": "土耳其",
        "arab": "阿拉伯地区",
        "west": "西方",
        "europe": "欧洲",
        "georgia": "格鲁吉亚",
        "germany": "德国",
        "anglo_pop": "英语流行",
        "kazakhstan": "哈萨克斯坦",
    }
    return culture_map.get(culture.lower(), humanize_title(region) or humanize_title(culture))


def translate_token(value: str) -> str:
    token = (value or "").strip().lower()
    mapping = {
        "voice": "人声",
        "instrumental": "器乐",
        "traditional": "传统",
        "zh": "中文",
        "en": "英文",
        "jingju acappella": "京剧清唱",
        "jingju_acappella": "京剧清唱",
        "traditional vocal": "传统声乐",
        "traditional_vocal": "传统声乐",
        "turkish makam": "土耳其马卡姆",
        "makam": "马卡姆",
        "china": "中国",
        "india": "印度",
        "germany": "德国",
        "kazakhstan": "哈萨克斯坦",
        "anglo pop": "英语流行",
        "anglo_pop": "英语流行",
    }
    return mapping.get(token, humanize_title(value))


def build_fallback_catalog(dim: int) -> list[dict[str, Any]]:
    seeds = [
        (
            "fallback_china",
            "京剧片段",
            "中国",
            "声腔纹理 / 戏曲装饰音",
            ["人声", "传统声乐", "戏曲"],
            92,
        ),
        (
            "fallback_turkey",
            "安纳托利亚鲁特琴",
            "土耳其",
            "拨弦音色 / 马卡姆旋律",
            ["拨弦", "马卡姆", "旋律"],
            104,
        ),
        (
            "fallback_india",
            "夜雨塔布拉",
            "印度",
            "鼓点脉冲 / 情绪推进",
            ["塔布拉", "节奏", "情绪"],
            118,
        ),
    ]
    items: list[dict[str, Any]] = []
    for track_id, title, origin, descriptor, tags, bpm in seeds:
        items.append(
            {
                "id": track_id,
                "title": title,
                "origin": origin,
                "culture": origin,
                "artist": "系统内置示例",
                "audio_path": "",
                "tags": tags,
                "descriptor": descriptor,
                "vector": deterministic_vector(f"{track_id}|{descriptor}|{origin}", dim),
                "bpm": bpm,
            }
        )
    return items


def mode_label(mode: str) -> str:
    return {
        "bridge": "桥接发现（Bridge）",
        "novelty": "新颖探索（Novelty）",
        "precision": "精准邻近（Precision）",
    }.get(mode, "桥接发现（Bridge）")


def lens_note(lens: str) -> str:
    return {
        "rhythm": "当前更强调节奏结构、律动形状和拍感邻近。",
        "timbre": "当前更强调音色包络、共振质感和乐器纹理。",
        "emotion": "当前更强调情绪轮廓、氛围张力和仪式感。",
    }.get(lens, "当前正在构建文化桥接解释。")


def clamp(value: float) -> float:
    return min(0.97, max(0.42, value))


def hash_string(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def seeded_value(seed: int) -> float:
    x = math.sin(seed) * 10000
    return x - math.floor(x)


def format_timestamp(timestamp_value: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_value))
