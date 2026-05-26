from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from dcas.ontology import OntologyStore

from .lightweight_catalog import get_lightweight_catalog
from .paths import Storage
from .prototype_api import create_prototype_router
from .schemas import (
    DatasetBuildRequest,
    KimiChatRequest,
    MainlineRecommendRequest,
    OntologyAnnotationCreateRequest,
    OntologyConceptCreateRequest,
    OntologyExportConstraintsRequest,
    OntologyRelationCreateRequest,
    OntologySuggestRequest,
    PalRequest,
    RecommendRequest,
    StyleTransferRequest,
    ToyGenerateRequest,
    TrainRequest,
    WaveStyleTransferRequest,
)


SUPPORTED_UPLOAD_AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".wav",
    ".wave",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".webm",
    ".aif",
    ".aiff",
    ".wma",
}

UPLOAD_ACCEPT_ATTRIBUTE = ",".join(
    [
        "audio/mpeg",
        "audio/mp4",
        "audio/aac",
        "audio/wav",
        "audio/x-wav",
        "audio/flac",
        "audio/ogg",
        "audio/opus",
        "audio/webm",
        *sorted(SUPPORTED_UPLOAD_AUDIO_EXTENSIONS),
    ]
)

DEFAULT_UPLOAD_COMPRESSION_BITRATE = "192k"
DEFAULT_UPLOAD_COMPRESSION_SAMPLE_RATE_HZ = 44_100
DEFAULT_UPLOAD_COMPRESSION_CHANNELS = 2
ALLOWED_UPLOAD_COMPRESSION_BITRATES = {"96k", "128k", "160k", "192k", "224k", "256k", "320k"}
ALLOWED_UPLOAD_COMPRESSION_SAMPLE_RATES = {24_000, 32_000, 44_100, 48_000}
ALLOWED_UPLOAD_COMPRESSION_CHANNELS = {1, 2}
ANON_SESSION_COOKIE = "echo_anon_id"
ANON_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
DEFAULT_INITIAL_FAVORITES = 20
WORKER_RELATIVE_URL_KEYS: set[str] = set()


def _track_key(track: dict[str, Any]) -> str:
    return str(
        track.get("track_id")
        or track.get("trackId")
        or track.get("id")
        or f"{track.get('title', '')}::{track.get('artist', '')}::{track.get('album', '')}"
    ).strip()


def _stable_seed(value: str) -> int:
    import hashlib

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _safe_anon_session(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return raw if ANON_SESSION_RE.match(raw) else uuid4().hex


class AnonymousFavoriteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anonymous_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS favorite_tracks (
                    session_id TEXT NOT NULL,
                    track_key TEXT NOT NULL,
                    track_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, track_key),
                    FOREIGN KEY (session_id) REFERENCES anonymous_sessions(session_id) ON DELETE CASCADE
                )
                """
            )

    def touch_session(self, session_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO anonymous_sessions (session_id, created_at, last_seen_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_seen_at=excluded.last_seen_at
                """,
                (session_id, now, now),
            )

    def list_favorites(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT track_json
                FROM favorite_tracks
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                item = json.loads(str(row["track_json"]))
            except Exception:
                continue
            if isinstance(item, dict):
                item["favorite"] = True
                items.append(item)
        return items

    def upsert_favorite(self, session_id: str, track: dict[str, Any]) -> dict[str, Any]:
        item = dict(track)
        key = _track_key(item)
        if not key:
            raise ValueError("favorite track requires an id, track_id, or title")
        item.setdefault("id", key)
        item.setdefault("track_id", key)
        item["favorite"] = True
        item["favorite_added_at"] = item.get("favorite_added_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        now = time.time()
        self.touch_session(session_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO favorite_tracks (session_id, track_key, track_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, track_key) DO UPDATE SET
                    track_json=excluded.track_json,
                    updated_at=excluded.updated_at
                """,
                (session_id, key, json.dumps(item, ensure_ascii=False), now, now),
            )
        return item

    def remove_favorite(self, session_id: str, track_key: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM favorite_tracks WHERE session_id = ? AND track_key = ?",
                (session_id, track_key),
            )
        return bool(cursor.rowcount)


def _cookie_secure_default() -> bool:
    return str(os.environ.get("ECHO_COOKIE_SECURE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _session_id_from_request(request: Request, response: Response, store: AnonymousFavoriteStore) -> str:
    previous = str(request.cookies.get(ANON_SESSION_COOKIE) or "").strip()
    session_id = _safe_anon_session(previous)
    if session_id != previous:
        response.set_cookie(
            ANON_SESSION_COOKIE,
            session_id,
            max_age=60 * 60 * 24 * 365 * 5,
            httponly=True,
            samesite="lax",
            secure=_cookie_secure_default(),
            path="/",
        )
    store.touch_session(session_id)
    return session_id


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _mainline_worker_url() -> str:
    return str(os.environ.get("ECHO_MAINLINE_WORKER_URL", "")).strip().rstrip("/")


def _mainline_worker_token() -> str:
    return str(os.environ.get("ECHO_MAINLINE_WORKER_TOKEN", "")).strip()


def _mainline_worker_timeout() -> float:
    try:
        return max(5.0, float(os.environ.get("ECHO_MAINLINE_WORKER_TIMEOUT_SECONDS", "900")))
    except Exception:
        return 900.0


def _worker_url(path: str, query: dict[str, Any] | None = None) -> str:
    base = _mainline_worker_url()
    if not base:
        raise HTTPException(status_code=503, detail="mainline worker is not configured")
    url = urljoin(f"{base}/", path.lstrip("/"))
    if query:
        clean_query = {key: value for key, value in query.items() if value is not None}
        if clean_query:
            url = f"{url}?{urlencode(clean_query)}"
    return url


def _worker_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = dict(extra or {})
    token = _mainline_worker_token()
    if token:
        headers["X-Echo-Worker-Token"] = token
    return headers


def _require_worker_token(request: Request) -> None:
    if not _env_bool("ECHO_WORKER_REQUIRE_TOKEN", False):
        return
    expected = str(os.environ.get("ECHO_WORKER_SHARED_TOKEN", "") or os.environ.get("ECHO_MAINLINE_WORKER_TOKEN", "")).strip()
    if not expected:
        raise HTTPException(status_code=500, detail="worker token enforcement is enabled but no token is configured")
    actual = str(request.headers.get("X-Echo-Worker-Token", "")).strip()
    if actual != expected:
        raise HTTPException(status_code=401, detail="invalid worker token")


def _rewrite_worker_urls(data: Any, worker_base: str) -> Any:
    if isinstance(data, list):
        return [_rewrite_worker_urls(item, worker_base) for item in data]
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in WORKER_RELATIVE_URL_KEYS and isinstance(value, str) and value.startswith("/"):
            out[key] = urljoin(f"{worker_base.rstrip('/')}/", value.lstrip("/"))
        else:
            out[key] = _rewrite_worker_urls(value, worker_base)
    return out


def _proxy_worker_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _worker_url(path),
        data=body,
        method="POST",
        headers=_worker_headers({"Content-Type": "application/json"}),
    )
    try:
        with urllib.request.urlopen(req, timeout=_mainline_worker_timeout()) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:4000]
        raise HTTPException(status_code=e.code, detail=detail)
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"mainline worker unavailable: {e.reason}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="mainline worker returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return _rewrite_worker_urls(data, _mainline_worker_url())


def _proxy_worker_get(path: str, query: dict[str, Any] | None = None) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        _worker_url(path, query),
        method="GET",
        headers=_worker_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=_mainline_worker_timeout()) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:4000]
        raise HTTPException(status_code=e.code, detail=detail)
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"mainline worker unavailable: {e.reason}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="mainline worker returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return _rewrite_worker_urls(data, _mainline_worker_url())


def _proxy_worker_upload_recommend(
    *,
    fields: dict[str, Any],
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    boundary = f"----echo-worker-{uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    safe_name = Path(filename or "uploaded_audio").name
    media_type = content_type or "application/octet-stream"
    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'.encode("utf-8"),
            f"Content-Type: {media_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    body = b"".join(parts)
    req = urllib.request.Request(
        _worker_url("/api/mainline/upload_recommend"),
        data=body,
        method="POST",
        headers=_worker_headers({"Content-Type": f"multipart/form-data; boundary={boundary}"}),
    )
    try:
        with urllib.request.urlopen(req, timeout=_mainline_worker_timeout()) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:4000]
        raise HTTPException(status_code=e.code, detail=detail)
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"mainline worker unavailable: {e.reason}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="mainline worker returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return _rewrite_worker_urls(data, _mainline_worker_url())


def _validate_upload_audio(filename: str, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in SUPPORTED_UPLOAD_AUDIO_EXTENSIONS:
        return suffix
    if str(content_type or "").lower().startswith("audio/"):
        return suffix or ".audio"
    allowed = ", ".join(sorted(SUPPORTED_UPLOAD_AUDIO_EXTENSIONS))
    raise HTTPException(status_code=415, detail=f"unsupported audio format. Supported extensions: {allowed}")


def _safe_compression_bitrate(value: str | None) -> str:
    raw = str(value or DEFAULT_UPLOAD_COMPRESSION_BITRATE).strip().lower()
    return raw if raw in ALLOWED_UPLOAD_COMPRESSION_BITRATES else DEFAULT_UPLOAD_COMPRESSION_BITRATE


def _safe_sample_rate(value: int | str | None) -> int:
    try:
        raw = int(value or DEFAULT_UPLOAD_COMPRESSION_SAMPLE_RATE_HZ)
    except Exception:
        return DEFAULT_UPLOAD_COMPRESSION_SAMPLE_RATE_HZ
    return raw if raw in ALLOWED_UPLOAD_COMPRESSION_SAMPLE_RATES else DEFAULT_UPLOAD_COMPRESSION_SAMPLE_RATE_HZ


def _safe_channels(value: int | str | None) -> int:
    try:
        raw = int(value or DEFAULT_UPLOAD_COMPRESSION_CHANNELS)
    except Exception:
        return DEFAULT_UPLOAD_COMPRESSION_CHANNELS
    return raw if raw in ALLOWED_UPLOAD_COMPRESSION_CHANNELS else DEFAULT_UPLOAD_COMPRESSION_CHANNELS


def _compress_audio_for_analysis(
    source: Path,
    target: Path,
    *,
    max_seconds: float,
    bitrate: str = DEFAULT_UPLOAD_COMPRESSION_BITRATE,
    sample_rate_hz: int = DEFAULT_UPLOAD_COMPRESSION_SAMPLE_RATE_HZ,
    channels: int = DEFAULT_UPLOAD_COMPRESSION_CHANNELS,
) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    bitrate = _safe_compression_bitrate(bitrate)
    sample_rate_hz = _safe_sample_rate(sample_rate_hz)
    channels = _safe_channels(channels)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    if float(max_seconds) > 0:
        cmd.extend(["-t", f"{float(max_seconds):.6f}"])
    cmd.extend(["-vn", "-ac", str(channels), "-ar", str(sample_rate_hz), "-b:a", str(bitrate), str(target)])
    timeout = max(90.0, float(max_seconds or 0) + 60.0)
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg compression failed: {err}")
    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError("ffmpeg compression produced no output")
    raw_size = int(source.stat().st_size) if source.exists() else 0
    compressed_size = int(target.stat().st_size)
    return {
        "status": "compressed",
        "codec": "mp3",
        "sample_rate_hz": int(sample_rate_hz),
        "channels": int(channels),
        "bitrate": str(bitrate),
        "max_seconds": float(max_seconds),
        "raw_size_bytes": raw_size,
        "compressed_size_bytes": compressed_size,
        "ratio": float(compressed_size / raw_size) if raw_size else None,
    }


def _flatten_audio_tags(data: dict) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in [data.get("format") or {}, *(data.get("streams") or [])]:
        raw_tags = item.get("tags") or {}
        if not isinstance(raw_tags, dict):
            continue
        for key, value in raw_tags.items():
            text = str(value or "").strip()
            if text and key:
                tags.setdefault(str(key).lower(), text)
    return tags


def _probe_audio_tags(path: Path) -> dict[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "{}")
    except Exception:
        return {}
    return _flatten_audio_tags(data if isinstance(data, dict) else {})


def _tag_first(tags: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(tags.get(name.lower()) or "").strip()
        if value:
            return value
    return ""


def _extract_embedded_cover(source: Path, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        str(target),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=45)
    except Exception:
        return False
    return proc.returncode == 0 and target.exists() and target.stat().st_size > 0


def _upload_place_from_tags(tags: dict[str, str]) -> dict[str, object]:
    country = _tag_first(tags, "country", "location", "artist_location", "com.apple.iTunes.location")
    if country:
        return {
            "country": country,
            "city": _tag_first(tags, "city", "venue") or country,
            "lat": None,
            "lng": None,
            "location_precision": "音频标签位置",
            "location_note": "来自音频文件内嵌 metadata",
        }
    return {
        "country": "神秘的地方",
        "city": "太平洋中部",
        "lat": 0.0,
        "lng": -160.0,
        "location_precision": "太平洋默认坐标",
        "location_note": "上传音频没有提供地区信息，默认标注为来自神秘的地方",
    }


def _load_local_kimi_config() -> dict[str, str]:
    paths = [
        Path("configs/secrets/kimi.local.json"),
        Path("storage/secrets/kimi.local.json"),
    ]
    config: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        api_key = str(data.get("api_key") or data.get("apiKey") or "").strip()
        if api_key:
            config["api_key"] = api_key
            config["source"] = str(path)
        model = str(data.get("model") or "").strip()
        endpoint = str(data.get("endpoint") or "").strip()
        if model:
            config["model"] = model
        if endpoint:
            config["endpoint"] = endpoint
        if config.get("api_key"):
            break
    import os

    env_key = str(os.environ.get("KIMI_API_KEY", "")).strip()
    if env_key:
        config["api_key"] = env_key
        config["source"] = "env:KIMI_API_KEY"
    env_model = str(os.environ.get("KIMI_MODEL", "")).strip()
    env_endpoint = str(os.environ.get("KIMI_ENDPOINT", "")).strip()
    if env_model:
        config["model"] = env_model
    if env_endpoint:
        config["endpoint"] = env_endpoint
    return config


def create_app() -> FastAPI:
    app = FastAPI(title="DCAS API", version="0.1.0")

    @app.middleware("http")
    async def worker_token_guard(request: Request, call_next):
        if _env_bool("ECHO_WORKER_REQUIRE_TOKEN", False):
            if not request.url.path.startswith("/api/mainline"):
                return JSONResponse(status_code=404, content={"detail": "worker mode only exposes mainline API"})
            expected = str(os.environ.get("ECHO_WORKER_SHARED_TOKEN", "") or os.environ.get("ECHO_MAINLINE_WORKER_TOKEN", "")).strip()
            actual = str(request.headers.get("X-Echo-Worker-Token", "")).strip()
            if not expected:
                return JSONResponse(status_code=500, content={"detail": "worker token enforcement is enabled but no token is configured"})
            if actual != expected:
                return JSONResponse(status_code=401, content={"detail": "invalid worker token"})
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    storage = Storage(root=Path(os.environ.get("ECHO_STORAGE_ROOT", "storage")))
    storage.ensure_dir("datasets")
    storage.ensure_dir("models")
    storage.ensure_dir("uploads")
    storage.ensure_dir("pal")
    storage.ensure_dir("style")
    storage.ensure_dir("ontology")
    storage.ensure_dir("prototype")
    ontology = OntologyStore(storage.resolve_rel("ontology/state.json"))
    user_data_root = Path(os.environ.get("ECHO_USER_DATA_DIR", str(storage.resolve_rel("user_data"))))
    favorite_store = AnonymousFavoriteStore(user_data_root / "echo.sqlite3")
    app.include_router(create_prototype_router(storage))

    def seed_initial_favorites(session_id: str, *, prefer_cuda: bool = False) -> tuple[list[dict[str, Any]], bool]:
        existing = favorite_store.list_favorites(session_id)
        if existing:
            return existing, False
        try:
            catalog = get_lightweight_catalog(storage)
            catalog_result = catalog.catalog(
                limit=DEFAULT_INITIAL_FAVORITES,
                random_seed=_stable_seed(session_id),
                exclude_low_signal=True,
            )
        except Exception:
            return [], False
        items = catalog_result.get("items") or []
        seeded: list[dict[str, Any]] = []
        for item in items[:DEFAULT_INITIAL_FAVORITES]:
            if not isinstance(item, dict):
                continue
            seeded.append(favorite_store.upsert_favorite(session_id, item))
        return favorite_store.list_favorites(session_id), bool(seeded)

    @app.get("/api/health")
    def health():
        return {"ok": True, "time": time.time()}

    @app.get("/api/session")
    def api_session(request: Request, response: Response):
        session_id = _session_id_from_request(request, response, favorite_store)
        return {"ok": True, "session_id": session_id, "cookie": ANON_SESSION_COOKIE}

    @app.get("/api/favorites")
    def api_list_favorites(
        request: Request,
        response: Response,
        seed: bool = True,
        prefer_cuda: bool = False,
    ):
        session_id = _session_id_from_request(request, response, favorite_store)
        if seed:
            items, seeded = seed_initial_favorites(session_id, prefer_cuda=prefer_cuda)
        else:
            items, seeded = favorite_store.list_favorites(session_id), False
        return {"ok": True, "session_id": session_id, "count": len(items), "seeded": seeded, "items": items}

    @app.post("/api/favorites")
    def api_add_favorite(payload: dict[str, Any], request: Request, response: Response):
        session_id = _session_id_from_request(request, response, favorite_store)
        track = payload.get("track") if isinstance(payload.get("track"), dict) else payload
        try:
            item = favorite_store.upsert_favorite(session_id, track)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True, "session_id": session_id, "item": item, "items": favorite_store.list_favorites(session_id)}

    @app.delete("/api/favorites")
    def api_remove_favorite_by_query(track_key: str, request: Request, response: Response):
        session_id = _session_id_from_request(request, response, favorite_store)
        removed = favorite_store.remove_favorite(session_id, track_key)
        return {"ok": True, "session_id": session_id, "removed": removed, "items": favorite_store.list_favorites(session_id)}

    @app.delete("/api/favorites/{track_key:path}")
    def api_remove_favorite(track_key: str, request: Request, response: Response):
        session_id = _session_id_from_request(request, response, favorite_store)
        removed = favorite_store.remove_favorite(session_id, track_key)
        return {"ok": True, "session_id": session_id, "removed": removed, "items": favorite_store.list_favorites(session_id)}

    @app.get("/api/files")
    def list_files():
        root = storage.root.resolve()
        files: list[str] = []
        for p in root.rglob("*"):
            if p.is_file():
                files.append(storage.relpath(p))
        files.sort()
        return {"files": files}

    @app.get("/api/files/download")
    def download(path: str):
        try:
            p = storage.resolve_rel(path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(p))

    @app.post("/api/files/upload")
    async def upload(file: UploadFile = File(...), dir: str = "uploads"):
        try:
            target_dir = storage.ensure_dir(dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid dir")
        name = Path(file.filename or "file.bin").name
        dest = (target_dir / name).resolve()
        if storage.root.resolve() not in dest.parents:
            raise HTTPException(status_code=400, detail="invalid filename")
        content = await file.read()
        dest.write_bytes(content)
        return {"path": storage.relpath(dest), "size": int(len(content))}

    @app.post("/api/toy/generate")
    def api_generate_toy(req: ToyGenerateRequest):
        from dcas.pipelines import generate_toy

        dataset_dir = storage.ensure_dir(f"datasets/{req.name}")
        out = generate_toy(out_dir=dataset_dir, n_tracks=req.n_tracks, dim=req.dim, seed=req.seed)
        return {
            "dir": storage.relpath(Path(out["dir"])),
            "tracks": storage.relpath(Path(out["tracks"])),
            "interactions": storage.relpath(Path(out["interactions"])),
            "meta": storage.relpath(Path(out["meta"])),
        }

    @app.post("/api/dataset/build_from_audio")
    def api_build_dataset(req: DatasetBuildRequest):
        from dcas.pipelines import build_tracks_with_culturemert

        try:
            metadata_path = storage.resolve_rel(req.metadata_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="metadata not found")
        out_path = storage.resolve_rel(f"datasets/{Path(req.out_name).name}")
        result = build_tracks_with_culturemert(
            metadata_csv=str(metadata_path),
            out_tracks_path=str(out_path),
            model_id=req.model_id,
            device=req.device,
            pooling=req.pooling,
            max_seconds=req.max_seconds,
            limit=req.limit,
            skip_errors=req.skip_errors,
        )
        result["out"] = storage.relpath(Path(str(result["out"])))
        if "manifest" in result:
            result["manifest"] = storage.relpath(Path(str(result["manifest"])))
        return result

    @app.post("/api/train")
    def api_train(req: TrainRequest):
        from dcas.pipelines import train_model

        try:
            tracks_path = storage.resolve_rel(req.tracks_path)
            constraints_path = storage.resolve_rel(req.constraints_path) if req.constraints_path else None
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")
        if constraints_path is not None and not constraints_path.exists():
            raise HTTPException(status_code=404, detail="constraints not found")

        out_path = storage.resolve_rel(f"models/{Path(req.out_name).name}")
        result = train_model(
            tracks_path=str(tracks_path),
            out_path=str(out_path),
            constraints_path=str(constraints_path) if constraints_path else None,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            seed=req.seed,
            prefer_cuda=req.prefer_cuda,
            lambda_constraints=req.lambda_constraints,
            constraint_margin=req.constraint_margin,
            lambda_domain=req.lambda_domain,
            lambda_contrast=req.lambda_contrast,
            lambda_cov=req.lambda_cov,
            lambda_tc=req.lambda_tc,
            lambda_hsic=req.lambda_hsic,
            beta_kl=req.beta_kl,
            shared_encoder=req.shared_encoder,
            regularizer_warmup_epochs=req.regularizer_warmup_epochs,
        )
        result["checkpoint"] = storage.relpath(Path(result["checkpoint"]))
        return result

    @app.post("/api/recommend")
    def api_recommend(req: RecommendRequest):
        from dcas.pipelines import recommend

        try:
            model_path = storage.resolve_rel(req.model_path)
            tracks_path = storage.resolve_rel(req.tracks_path)
            interactions_path = storage.resolve_rel(req.interactions_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")
        if not interactions_path.exists():
            raise HTTPException(status_code=404, detail="interactions not found")

        return recommend(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            interactions_path=str(interactions_path),
            user_id=req.user_id,
            target_culture=req.target_culture,
            k=req.k,
            prefer_cuda=req.prefer_cuda,
            epsilon=req.epsilon,
            iters=req.iters,
        )

    @app.get("/api/mainline/status")
    def api_mainline_status(prefer_cuda: bool = False):
        if _mainline_worker_url():
            try:
                data = _proxy_worker_get("/api/mainline/status", {"prefer_cuda": prefer_cuda})
                data["worker"] = {"configured": True, "online": True, "url": _mainline_worker_url()}
                return data
            except HTTPException as e:
                try:
                    data = get_lightweight_catalog(storage).status()
                except Exception:
                    raise e
                data["worker"] = {
                    "configured": True,
                    "online": False,
                    "url": _mainline_worker_url(),
                    "error": str(e.detail),
                }
                return data
        try:
            data = get_lightweight_catalog(storage).status()
            data["worker"] = {"configured": False, "online": False, "url": ""}
            return data
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/cultures")
    def api_mainline_cultures(prefer_cuda: bool = False):
        try:
            return get_lightweight_catalog(storage).cultures()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/catalog")
    def api_mainline_catalog(
        culture: str | None = None,
        source_dataset: str | None = None,
        q: str | None = None,
        limit: int = 24,
        random_seed: int | None = 42,
        exclude_low_signal: bool = True,
        prefer_cuda: bool = False,
    ):
        try:
            catalog = get_lightweight_catalog(storage)
            return catalog.catalog(
                culture=culture,
                source_dataset=source_dataset,
                q=q,
                limit=limit,
                random_seed=random_seed,
                exclude_low_signal=exclude_low_signal,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/random")
    def api_mainline_random(
        culture: str | None = None,
        source_dataset: str | None = None,
        random_seed: int | None = 42,
        exclude_low_signal: bool = True,
        prefer_cuda: bool = False,
    ):
        try:
            catalog = get_lightweight_catalog(storage)
            return catalog.random_track(
                culture=culture,
                source_dataset=source_dataset,
                random_seed=random_seed,
                exclude_low_signal=exclude_low_signal,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/tracks/{track_id}")
    def api_mainline_track(track_id: str, prefer_cuda: bool = False):
        try:
            return get_lightweight_catalog(storage).track(track_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/audio/{track_id}")
    def api_mainline_audio(track_id: str, prefer_cuda: bool = False):
        try:
            catalog = get_lightweight_catalog(storage)
            path, media_type = catalog.audio_file(track_id)
            return FileResponse(str(path), media_type=media_type, headers={"Accept-Ranges": "bytes"})
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except FileNotFoundError:
            try:
                track = get_lightweight_catalog(storage).track(track_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
            preview_url = str(track.get("preview_url") or "").strip()
            if preview_url.startswith(("http://", "https://")):
                return RedirectResponse(preview_url)
            if _mainline_worker_url():
                return RedirectResponse(_worker_url(f"/api/mainline/audio/{track_id}", {"prefer_cuda": prefer_cuda}))
            raise HTTPException(status_code=404, detail="audio not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/mainline/recommend")
    def api_mainline_recommend(req: MainlineRecommendRequest, request: Request):
        _require_worker_token(request)
        if _mainline_worker_url():
            return _proxy_worker_json("/api/mainline/recommend", req.dict())
        from .mainline_platform import MainlineWeights, get_mainline_platform

        seed_track_ids = list(req.seed_track_ids)
        if req.seed_track_id:
            seed_track_ids.insert(0, req.seed_track_id)
        weights = MainlineWeights(
            relevance=req.relevance_weight,
            novelty=req.novelty_weight,
            target_affinity=req.target_affinity_weight,
            minority=req.minority_weight,
            source=req.source_weight,
            diversity_lambda=req.diversity_lambda,
        )
        try:
            platform = get_mainline_platform(storage, prefer_cuda=req.prefer_cuda)
            return platform.recommend(
                seed_track_ids=seed_track_ids,
                seed_culture=req.seed_culture,
                target_culture=req.target_culture,
                mode=req.mode,
                k=req.k,
                recall_k=req.recall_k,
                random_seed=req.random_seed,
                exclude_same_artist=req.exclude_same_artist,
                exclude_low_signal=req.exclude_low_signal,
                weights=weights,
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/upload_formats")
    def api_mainline_upload_formats():
        return {
            "ok": True,
            "mode": "worker" if _mainline_worker_url() else "local",
            "worker_configured": bool(_mainline_worker_url()),
            "extensions": sorted(SUPPORTED_UPLOAD_AUDIO_EXTENSIONS),
            "accept": UPLOAD_ACCEPT_ATTRIBUTE,
            "max_upload_mb": 200,
            "default_compression": {
                "enabled": True,
                "codec": "mp3",
                "sample_rate_hz": DEFAULT_UPLOAD_COMPRESSION_SAMPLE_RATE_HZ,
                "channels": DEFAULT_UPLOAD_COMPRESSION_CHANNELS,
                "bitrate": DEFAULT_UPLOAD_COMPRESSION_BITRATE,
                "trimmed_to_analysis_window": True,
                "analysis_only": True,
                "raw_upload_preserved_for_playback_and_metadata": True,
            },
        }

    @app.post("/api/mainline/upload_recommend")
    async def api_mainline_upload_recommend(
        request: Request,
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        artist: str | None = Form(default=None),
        seed_culture: str | None = Form(default=None),
        target_culture: str | None = Form(default=None),
        mode: str = Form(default="open"),
        k: int = Form(default=10),
        recall_k: int = Form(default=900),
        random_seed: int | None = Form(default=42),
        prefer_cuda: bool = Form(default=False),
        exclude_same_artist: bool = Form(default=False),
        exclude_low_signal: bool = Form(default=True),
        max_seconds: float = Form(default=30.0),
        window_count: int = Form(default=1),
        window_strategy: str = Form(default="single"),
        window_aggregate: str = Form(default="mean"),
        compress_upload: bool = Form(default=True),
        compression_bitrate: str = Form(default=DEFAULT_UPLOAD_COMPRESSION_BITRATE),
        compression_sample_rate_hz: int = Form(default=DEFAULT_UPLOAD_COMPRESSION_SAMPLE_RATE_HZ),
        compression_channels: int = Form(default=DEFAULT_UPLOAD_COMPRESSION_CHANNELS),
    ):
        filename = Path(file.filename or "uploaded_audio").name
        suffix = _validate_upload_audio(filename, file.content_type)
        saved_stem = f"{uuid4().hex}_{Path(filename).stem[:80] or 'audio'}"
        saved_name = f"{saved_stem}{suffix[:12]}"
        raw_dir = storage.ensure_dir("uploads/mainline/raw")
        compressed_dir = storage.ensure_dir("uploads/mainline/compressed")
        dest = (raw_dir / saved_name).resolve()
        max_bytes = 200 * 1024 * 1024
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="uploaded audio is larger than 200MB")
        if _mainline_worker_url():
            return _proxy_worker_upload_recommend(
                fields={
                    "title": title,
                    "artist": artist,
                    "seed_culture": seed_culture,
                    "target_culture": target_culture,
                    "mode": mode,
                    "k": k,
                    "recall_k": recall_k,
                    "random_seed": random_seed,
                    "prefer_cuda": prefer_cuda,
                    "exclude_same_artist": exclude_same_artist,
                    "exclude_low_signal": exclude_low_signal,
                    "max_seconds": max_seconds,
                    "window_count": window_count,
                    "window_strategy": window_strategy,
                    "window_aggregate": window_aggregate,
                    "compress_upload": compress_upload,
                    "compression_bitrate": compression_bitrate,
                    "compression_sample_rate_hz": compression_sample_rate_hz,
                    "compression_channels": compression_channels,
                },
                file_bytes=content,
                filename=filename,
                content_type=file.content_type,
            )
        _require_worker_token(request)
        from .mainline_platform import MainlineWeights, get_mainline_platform

        dest.write_bytes(content)

        weights = MainlineWeights()
        rel_upload_path = storage.relpath(dest)
        analysis_path = dest
        playback_path = dest
        tag_info = _probe_audio_tags(dest)
        cover_rel_path = ""
        cover_path = storage.ensure_dir("uploads/mainline/covers") / f"{saved_stem}.jpg"
        if _extract_embedded_cover(dest, cover_path):
            cover_rel_path = storage.relpath(cover_path)
        place_info = _upload_place_from_tags(tag_info)
        compression_info: dict[str, object] = {
            "status": "disabled",
            "raw_path": rel_upload_path,
            "raw_size_bytes": int(len(content)),
            "metadata_preserved_in_raw": True,
        }
        if bool(compress_upload):
            compressed_path = (compressed_dir / f"{saved_stem}.mp3").resolve()
            try:
                compression_info = _compress_audio_for_analysis(
                    dest,
                    compressed_path,
                    max_seconds=max(1.0, float(max_seconds)),
                    bitrate=compression_bitrate,
                    sample_rate_hz=compression_sample_rate_hz,
                    channels=compression_channels,
                )
                compression_info["path"] = storage.relpath(compressed_path)
                compression_info["raw_path"] = rel_upload_path
                compression_info["playback_path"] = rel_upload_path
                compression_info["analysis_only"] = True
                compression_info["metadata_preserved_in_raw"] = True
                analysis_path = compressed_path
                playback_path = dest
                compression_info["raw_deleted"] = False
            except Exception as e:
                compression_info = {
                    "status": "fallback_original",
                    "raw_path": rel_upload_path,
                    "raw_size_bytes": int(len(content)),
                    "metadata_preserved_in_raw": True,
                    "error": str(e)[:1200],
                }

        rel_playback_path = storage.relpath(playback_path)
        upload_info = {
            "track_id": f"upload_{dest.stem[:48]}",
            "filename": filename,
            "title": title or _tag_first(tag_info, "title") or Path(filename).stem,
            "artist": artist or _tag_first(tag_info, "artist", "album_artist", "albumartist", "composer") or "Uploaded audio",
            "album": _tag_first(tag_info, "album"),
            "genre": _tag_first(tag_info, "genre"),
            "release_date": _tag_first(tag_info, "date", "year"),
            "description": _tag_first(tag_info, "description", "comment", "synopsis"),
            "path": rel_playback_path,
            "analysis_path": storage.relpath(analysis_path),
            "size_bytes": int(len(content)),
            "content_type": file.content_type or "",
            "audio_api_url": f"/api/files/download?path={rel_playback_path}",
            "cover_art_url": f"/api/files/download?path={cover_rel_path}" if cover_rel_path else "",
            "cover_art_url_large": f"/api/files/download?path={cover_rel_path}" if cover_rel_path else "",
            "embedded_tags": {key: tag_info[key] for key in sorted(tag_info)[:80]},
            **place_info,
            "compression": compression_info,
        }
        try:
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda)
            result = platform.recommend_audio_file(
                audio_path=analysis_path,
                upload_info=upload_info,
                seed_culture=seed_culture,
                target_culture=target_culture,
                mode=mode,
                k=k,
                recall_k=recall_k,
                random_seed=random_seed,
                exclude_same_artist=exclude_same_artist,
                exclude_low_signal=exclude_low_signal,
                weights=weights,
                max_seconds=max_seconds,
                window_count=window_count,
                window_strategy=window_strategy,
                window_aggregate=window_aggregate,
            )
            return result
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/ai/kimi/status")
    def api_kimi_status():
        local = _load_local_kimi_config()
        return {
            "ok": True,
            "has_local_key": bool(local.get("api_key")),
            "source": local.get("source", ""),
            "model": local.get("model", "kimi-k2.6"),
            "endpoint": local.get("endpoint", "https://api.moonshot.cn/v1/chat/completions"),
        }

    @app.post("/api/ai/kimi/chat")
    def api_kimi_chat(req: KimiChatRequest):
        local = _load_local_kimi_config()
        api_key = str(req.api_key or "").strip() or str(local.get("api_key") or "").strip()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Kimi API key is not configured. Set it in settings.html, KIMI_API_KEY, or configs/secrets/kimi.local.json.",
            )
        endpoint = (req.endpoint.strip() if req.endpoint else "") or local.get("endpoint", "https://api.moonshot.cn/v1/chat/completions")
        if not endpoint.startswith("https://"):
            raise HTTPException(status_code=400, detail="Kimi endpoint must use https")
        model = (req.model.strip() if req.model else "") or local.get("model", "kimi-k2.6")
        payload = {
            "model": model,
            "messages": req.messages,
            "max_completion_tokens": int(req.max_completion_tokens),
            "thinking": {"type": "disabled"},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            import urllib.error
            import urllib.request

            upstream = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(upstream, timeout=float(req.timeout_seconds)) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:4000]
            raise HTTPException(status_code=e.code, detail=detail)
        except urllib.error.URLError as e:
            raise HTTPException(status_code=502, detail=str(e.reason))
        except TimeoutError:
            raise HTTPException(status_code=504, detail=f"Kimi upstream timed out after {float(req.timeout_seconds):.0f}s")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Kimi returned invalid JSON")
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        if not content:
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length" and message.get("reasoning_content"):
                raise HTTPException(
                    status_code=502,
                    detail="Kimi reasoning started but max_completion_tokens was too low to produce final content.",
                )
            raise HTTPException(status_code=502, detail="Kimi returned no message content")
        return {"ok": True, "content": content, "raw": data}

    @app.post("/api/style/transfer")
    def api_style_transfer(req: StyleTransferRequest):
        from dcas.pipelines import style_transfer

        try:
            model_path = storage.resolve_rel(req.model_path)
            tracks_path = storage.resolve_rel(req.tracks_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")

        out_path = storage.resolve_rel(f"style/{Path(req.out_name).name}")
        out = style_transfer(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            source_track_id=req.source_track_id,
            style_track_id=req.style_track_id,
            out_path=str(out_path),
            target_culture=req.target_culture,
            alpha=req.alpha,
            k=req.k,
            prefer_cuda=req.prefer_cuda,
        )
        out["artifact"] = storage.relpath(Path(str(out["artifact"])))
        return out

    @app.post("/api/style/transfer_waveform")
    def api_style_transfer_waveform(req: WaveStyleTransferRequest):
        from dcas.pipelines import style_transfer_waveform

        try:
            source_audio = storage.resolve_rel(req.source_audio_path)
            style_audio = storage.resolve_rel(req.style_audio_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not source_audio.exists():
            raise HTTPException(status_code=404, detail="source audio not found")
        if not style_audio.exists():
            raise HTTPException(status_code=404, detail="style audio not found")

        out_path = storage.resolve_rel(f"style/{Path(req.out_name).name}")
        out = style_transfer_waveform(
            source_audio_path=str(source_audio),
            style_audio_path=str(style_audio),
            out_wav_path=str(out_path),
            alpha=req.alpha,
            target_sr=req.target_sr,
            n_fft=req.n_fft,
            hop_length=req.hop_length,
            win_length=req.win_length,
            max_seconds=req.max_seconds,
            peak_norm=req.peak_norm,
        )
        out["artifact"] = storage.relpath(Path(str(out["artifact"])))
        out["source_audio_path"] = storage.relpath(source_audio)
        out["style_audio_path"] = storage.relpath(style_audio)
        return out

    @app.post("/api/pal")
    def api_pal(req: PalRequest):
        from dcas.pipelines import pal_tasks

        try:
            model_path = storage.resolve_rel(req.model_path)
            tracks_path = storage.resolve_rel(req.tracks_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="model not found")
        if not tracks_path.exists():
            raise HTTPException(status_code=404, detail="tracks not found")
        out_path = storage.resolve_rel(f"pal/{Path(req.out_name).name}")
        result = pal_tasks(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            out_path=str(out_path),
            n=req.n,
            prefer_cuda=req.prefer_cuda,
        )
        result["tasks"] = storage.relpath(Path(result["tasks"]))
        return result

    @app.get("/api/ontology/state")
    def api_ontology_state():
        return ontology.state()

    @app.post("/api/ontology/concepts")
    def api_ontology_add_concept(req: OntologyConceptCreateRequest):
        try:
            obj = ontology.add_concept(
                name=req.name,
                description=req.description,
                parent_id=req.parent_id,
                aliases=req.aliases,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obj

    @app.post("/api/ontology/relations")
    def api_ontology_add_relation(req: OntologyRelationCreateRequest):
        try:
            obj = ontology.add_relation(
                source_id=req.source_id,
                target_id=req.target_id,
                relation_type=req.relation_type,
                weight=req.weight,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obj

    @app.post("/api/ontology/annotations")
    def api_ontology_add_annotation(req: OntologyAnnotationCreateRequest):
        try:
            obj = ontology.add_annotation(
                track_id=req.track_id,
                concept_id=req.concept_id,
                confidence=req.confidence,
                source=req.source,
                rationale=req.rationale,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obj

    @app.post("/api/ontology/suggest")
    def api_ontology_suggest(req: OntologySuggestRequest):
        return {"items": ontology.suggest_concepts(query=req.query, top_k=req.top_k)}

    @app.post("/api/ontology/export_constraints")
    def api_ontology_export_constraints(req: OntologyExportConstraintsRequest):
        out_path = storage.resolve_rel(f"ontology/{Path(req.out_name).name}")
        result = ontology.save_pairwise_constraints(
            path=out_path,
            min_confidence=req.min_confidence,
            max_pairs_per_concept=req.max_pairs_per_concept,
        )
        result["path"] = storage.relpath(Path(str(result["path"])))
        return result

    # --- PAL Annotation Web UI endpoints ---

    @app.get("/api/pal/tasks")
    def get_pal_tasks():
        """Load enriched PAL tasks JSON."""
        tasks_path = Path("storage/pal/v4_main_annotation/pal_tasks.json")
        if not tasks_path.exists():
            raise HTTPException(status_code=404, detail="PAL tasks not found. Run pal_tasks first.")
        with open(tasks_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    @app.get("/api/pal/audio")
    def stream_audio(path: str):
        """Stream an audio file by absolute or relative path."""
        p = Path(path)
        if not p.is_absolute():
            p = Path("E:/Desktop/Echo") / p
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail=f"audio not found: {path}")
        ext = p.suffix.lower()
        media_type = "audio/mpeg" if ext == ".mp3" else "audio/wav" if ext == ".wav" else "audio/octet-stream"
        return FileResponse(str(p), media_type=media_type, headers={"Accept-Ranges": "bytes"})

    @app.post("/api/pal/annotate")
    def save_annotation(ann: dict):
        """Save a single annotation to JSONL file."""
        out_dir = Path("storage/pal/v4_main_annotation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "annotations.jsonl"
        record = {
            "task_id": ann.get("task_id", ""),
            "track_id_a": ann.get("track_id_a", ""),
            "track_id_b": ann.get("track_id_b", ""),
            "judgment": ann.get("judgment", ""),  # "a", "b", or "neither"
            "rationale": ann.get("rationale", ""),
            "annotator": ann.get("annotator", "web"),
            "timestamp": time.time(),
        }
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "saved": int(out_file.stat().st_size > 0)}

    @app.get("/api/pal/progress")
    def get_pal_progress():
        """Return how many annotations have been saved."""
        out_file = Path("storage/pal/v4_main_annotation/annotations.jsonl")
        if not out_file.exists():
            return {"count": 0, "total": 60}
        with open(out_file, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        return {"count": count, "total": 60}

    @app.post("/api/pal/export")
    def export_annotations():
        """Export annotations as CSV for constraint building."""
        out_file = Path("storage/pal/v4_main_annotation/annotations.jsonl")
        csv_file = Path("storage/pal/v4_main_annotation/annotated.csv")
        if not out_file.exists():
            raise HTTPException(status_code=404, detail="no annotations yet")

        import csv as _csv
        # Load annotations
        anns = []
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    anns.append(json.loads(line))

        # Load tasks for enrichment
        tasks_map = {}
        tasks_json = Path("storage/pal/v4_main_annotation/pal_tasks.json")
        if tasks_json.exists():
            with open(tasks_json, "r", encoding="utf-8") as f:
                for t in json.load(f):
                    tasks_map[(t["track_id_a"], t["track_id_b"])] = t

        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["task_id", "track_id_a", "track_id_b", "judgment", "similar", "rationale", "annotator"])
            for a in anns:
                judgment = a.get("judgment", "")
                similar = "yes" if judgment in ("a", "b") else ("no" if judgment == "neither" else "")
                writer.writerow([
                    a["task_id"], a["track_id_a"], a["track_id_b"],
                    judgment, similar, a.get("rationale", ""), a.get("annotator", ""),
                ])
        return {"csv": str(csv_file), "count": len(anns)}

    web_dir = Path("web")
    dist = Path("web/dist")
    prototype_dir = Path("web_prototype")
    if prototype_dir.exists():
        app.mount("/prototype", StaticFiles(directory=str(prototype_dir), html=True), name="prototype")
    # PAL annotation UI
    pal_html = dist / "pal.html"
    if pal_html.exists():
        app.mount("/pal", StaticFiles(directory=str(pal_html.parent), html=False), name="pal")
    if (web_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    elif (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
    elif prototype_dir.exists():
        app.mount("/", StaticFiles(directory=str(prototype_dir), html=True), name="prototype-root")
    else:
        dist.mkdir(parents=True, exist_ok=True)
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web-empty")

    return app


app = create_app()
