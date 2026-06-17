from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dcas.ontology import OntologyStore

from .lightweight_catalog import get_lightweight_catalog
from .paths import Storage
from .prototype_api import create_prototype_router
from .schemas import (
    DatasetBuildRequest,
    KimiChatRequest,
    KimiTrackCountryRequest,
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
ALLOWED_UPLOAD_COMPRESSION_BITRATES = {
    "96k",
    "128k",
    "160k",
    "192k",
    "224k",
    "256k",
    "320k",
}
ALLOWED_UPLOAD_COMPRESSION_SAMPLE_RATES = {24_000, 32_000, 44_100, 48_000}
ALLOWED_UPLOAD_COMPRESSION_CHANNELS = {1, 2}
ANON_SESSION_COOKIE = "echo_anon_id"
ANON_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
DEFAULT_INITIAL_FAVORITES = 20
WORKER_RELATIVE_URL_KEYS: set[str] = set()
DEFAULT_MAINLINE_TRACKS_REL = "public/merged/tracks_culturemert.npz"
DEFAULT_MAINLINE_METADATA_REL = "public/merged/metadata_merged.csv"
DEFAULT_MAINLINE_MODEL_REL = "models/dcas_full_v4_main_culturemert_stage3.pt"
DEFAULT_CULTUREMERT_MODEL_ID = "ntua-slp/CultureMERT-95M"
DEFAULT_UPLOAD_EMBEDDING_PROVIDER = "culturemert"
DEFAULT_GEMINI_EMBEDDING_MODEL_ID = "gemini-embedding-2"
DEFAULT_GEMINI_EMBEDDING_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _mainline_metadata_setting() -> str:
    return (
        os.environ.get("ECHO_MAINLINE_METADATA_PATH")
        or os.environ.get("ECHO_MAINLINE_METADATA_REL")
        or DEFAULT_MAINLINE_METADATA_REL
    )


def _mainline_tracks_setting() -> str:
    return (
        os.environ.get("ECHO_MAINLINE_TRACKS_PATH")
        or os.environ.get("ECHO_MAINLINE_TRACKS_REL")
        or DEFAULT_MAINLINE_TRACKS_REL
    )


def _mainline_model_setting() -> str:
    return (
        os.environ.get("ECHO_MAINLINE_MODEL_PATH")
        or os.environ.get("ECHO_MAINLINE_MODEL_REL")
        or DEFAULT_MAINLINE_MODEL_REL
    )


def _mainline_platform_paths() -> dict[str, str]:
    return {
        "tracks_rel": _mainline_tracks_setting(),
        "metadata_rel": _mainline_metadata_setting(),
        "model_rel": _mainline_model_setting(),
    }


def _culturemert_runtime_settings() -> dict[str, Any]:
    return {
        "culturemert_model_id": str(
            os.environ.get("ECHO_CULTUREMERT_MODEL_ID") or DEFAULT_CULTUREMERT_MODEL_ID
        ).strip(),
        "culturemert_cache_dir": str(os.environ.get("ECHO_CULTUREMERT_CACHE_DIR") or "").strip() or None,
        "culturemert_revision": str(os.environ.get("ECHO_CULTUREMERT_REVISION") or "").strip() or None,
        "culturemert_local_files_only": _env_bool("ECHO_CULTUREMERT_LOCAL_FILES_ONLY", False),
    }


def _upload_embedding_provider() -> str:
    raw = str(os.environ.get("ECHO_UPLOAD_EMBEDDING_PROVIDER") or DEFAULT_UPLOAD_EMBEDDING_PROVIDER).strip().lower()
    aliases = {
        "local": "culturemert",
        "culturemert": "culturemert",
        "gemini": "gemini",
        "gemini_embedding2": "gemini",
        "gemini-embedding-2": "gemini",
    }
    return aliases.get(raw, raw)


def _env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return int(value)


def _env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return float(value)


def _gemini_embedding_runtime_settings() -> dict[str, Any]:
    return {
        "api_key": str(os.environ.get("GEMINI_API_KEY") or os.environ.get("ECHO_GEMINI_API_KEY") or "").strip(),
        "model_id": str(
            os.environ.get("ECHO_GEMINI_EMBEDDING_MODEL")
            or os.environ.get("GEMINI_EMBEDDING_MODEL")
            or DEFAULT_GEMINI_EMBEDDING_MODEL_ID
        ).strip(),
        "api_base": str(os.environ.get("ECHO_GEMINI_EMBEDDING_API_BASE") or DEFAULT_GEMINI_EMBEDDING_API_BASE)
        .strip()
        .rstrip("/"),
        "output_dimensionality": _env_int("ECHO_GEMINI_EMBEDDING_DIM", 768, min_value=1, max_value=3072),
        "target_sample_rate": _env_int("ECHO_GEMINI_EMBEDDING_SAMPLE_RATE_HZ", 16000, min_value=8000, max_value=48000),
        "max_seconds": _env_float("ECHO_GEMINI_EMBEDDING_MAX_SECONDS", 30.0, min_value=1.0, max_value=180.0),
        "window_count": _env_int("ECHO_GEMINI_EMBEDDING_WINDOW_COUNT", 1, min_value=1, max_value=12),
        "window_strategy": str(os.environ.get("ECHO_GEMINI_EMBEDDING_WINDOW_STRATEGY") or "single").strip() or "single",
        "window_aggregate": str(os.environ.get("ECHO_GEMINI_EMBEDDING_WINDOW_AGGREGATE") or "mean").strip() or "mean",
        "request_timeout_s": _env_int("ECHO_GEMINI_EMBEDDING_TIMEOUT_SECONDS", 180, min_value=5, max_value=600),
        "max_retries": _env_int("ECHO_GEMINI_EMBEDDING_MAX_RETRIES", 5, min_value=1, max_value=20),
        "retry_backoff_s": _env_float(
            "ECHO_GEMINI_EMBEDDING_RETRY_BACKOFF_SECONDS", 2.0, min_value=0.1, max_value=60.0
        ),
        "audio_mime_type": str(os.environ.get("ECHO_GEMINI_EMBEDDING_AUDIO_MIME_TYPE") or "audio/wav").strip()
        or "audio/wav",
        "task_type": str(os.environ.get("ECHO_GEMINI_EMBEDDING_TASK_TYPE") or "").strip() or None,
        "title": str(os.environ.get("ECHO_GEMINI_EMBEDDING_TITLE") or "").strip() or None,
        "vertexai": _env_bool("ECHO_GEMINI_EMBEDDING_VERTEXAI", False),
        "vertex_project": str(os.environ.get("ECHO_GEMINI_VERTEX_PROJECT") or "").strip() or None,
        "vertex_location": str(os.environ.get("ECHO_GEMINI_VERTEX_LOCATION") or "").strip() or None,
    }


def _public_gemini_embedding_settings() -> dict[str, Any]:
    settings = _gemini_embedding_runtime_settings()
    public = {key: value for key, value in settings.items() if key != "api_key"}
    public["has_api_key"] = bool(settings.get("api_key"))
    return public


def _looks_like_culturemert_load_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    terms = (
        "culturemert",
        "huggingface",
        "from_pretrained",
        "processor_config",
        "preprocessor_config",
        "local_files_only",
        "ssl",
        "couldn't connect",
        "could not connect",
        "model id",
        "transformers",
    )
    return any(term in text for term in terms)


def _looks_like_gemini_embedding_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    terms = (
        "gemini",
        "google",
        "api key",
        "embedding",
        "generativelanguage",
        "vertex",
        "quota",
        "rate",
    )
    return any(term in text for term in terms)


def _embed_uploaded_audio_with_gemini(
    *,
    audio_path: Path,
    title: str,
    max_seconds: float | None,
    window_count: int,
    window_strategy: str,
    window_aggregate: str,
) -> tuple[Any, dict[str, Any]]:
    settings = _gemini_embedding_runtime_settings()
    if not settings["api_key"]:
        raise RuntimeError("GEMINI_API_KEY is required when ECHO_UPLOAD_EMBEDDING_PROVIDER=gemini")

    from dcas.embeddings.gemini_embedding2 import GeminiEmbedding2Config, GeminiEmbedding2Embedder

    started = time.time()
    cfg = GeminiEmbedding2Config(
        model_id=str(settings["model_id"]),
        api_key=str(settings["api_key"]),
        api_base=str(settings["api_base"]),
        vertexai=bool(settings["vertexai"]),
        vertex_project=settings["vertex_project"],
        vertex_location=settings["vertex_location"],
        output_dimensionality=int(settings["output_dimensionality"]),
        task_type=settings["task_type"],
        title=settings["title"],
        max_seconds=max_seconds if max_seconds is not None else float(settings["max_seconds"]),
        target_sample_rate=int(settings["target_sample_rate"]),
        window_count=int(window_count or settings["window_count"]),
        window_strategy=str(window_strategy or settings["window_strategy"]),
        window_aggregate=str(window_aggregate or settings["window_aggregate"]),
        request_timeout_s=int(settings["request_timeout_s"]),
        max_retries=int(settings["max_retries"]),
        retry_backoff_s=float(settings["retry_backoff_s"]),
        audio_mime_type=str(settings["audio_mime_type"]),
    )
    embedder = GeminiEmbedding2Embedder(cfg)
    emb, prep_report = embedder.embed_file(audio_path, title=title)
    meta = {
        "provider": "gemini",
        "model_id": str(cfg.model_id),
        "dim": int(emb.shape[0]),
        "output_dimensionality": int(cfg.output_dimensionality),
        "api_base": str(cfg.api_base),
        "vertexai": bool(cfg.vertexai),
        "max_seconds": cfg.max_seconds,
        "target_sample_rate": int(cfg.target_sample_rate),
        "window_count": int(cfg.window_count),
        "window_strategy": str(cfg.window_strategy),
        "window_aggregate": str(cfg.window_aggregate),
        "elapsed_seconds": float(time.time() - started),
        "preprocess": prep_report,
    }
    return emb, meta


def _warn_if_mainline_not_gemini(platform: Any) -> list[str]:
    text = " ".join(
        [
            str(getattr(platform, "tracks_path", "")),
            str(getattr(platform, "model_path", "")),
        ]
    ).lower()
    if "gemini" in text:
        return []
    return [
        "Gemini upload embeddings are enabled, but the configured mainline tracks/model names do not look Gemini-based. "
        "Use ECHO_MAINLINE_TRACKS_PATH and ECHO_MAINLINE_MODEL_PATH for Gemini-built artifacts to keep embedding spaces aligned."
    ]


def _normalize_kimi_thinking_mode(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return "thinking" if raw in {"thinking", "think", "reasoning", "slow"} else "fast"


def _kimi_thinking_config(mode: str) -> dict[str, str]:
    if _normalize_kimi_thinking_mode(mode) == "thinking":
        return {"type": "enabled"}
    return {"type": "disabled"}


def _kimi_request_payload(req: KimiChatRequest, model: str, *, stream: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": req.messages,
        "max_completion_tokens": int(req.max_completion_tokens),
        "thinking": _kimi_thinking_config(req.thinking_mode),
    }
    if stream:
        payload["stream"] = True
    return payload


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _string_delta(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


COUNTRY_CAPITALS: dict[str, dict[str, Any]] = {
    "US": {"country": "United States", "capital": "Washington, D.C.", "lat": 38.9072, "lng": -77.0369},
    "GB": {"country": "United Kingdom", "capital": "London", "lat": 51.5072, "lng": -0.1276},
    "CN": {"country": "China", "capital": "Beijing", "lat": 39.9042, "lng": 116.4074},
    "SG": {"country": "Singapore", "capital": "Singapore", "lat": 1.3521, "lng": 103.8198},
    "MY": {"country": "Malaysia", "capital": "Kuala Lumpur", "lat": 3.1390, "lng": 101.6869},
    "TH": {"country": "Thailand", "capital": "Bangkok", "lat": 13.7563, "lng": 100.5018},
    "VN": {"country": "Vietnam", "capital": "Hanoi", "lat": 21.0278, "lng": 105.8342},
    "PH": {"country": "Philippines", "capital": "Manila", "lat": 14.5995, "lng": 120.9842},
    "TW": {"country": "Taiwan", "capital": "Taipei", "lat": 25.0330, "lng": 121.5654},
    "HK": {"country": "Hong Kong", "capital": "Hong Kong", "lat": 22.3193, "lng": 114.1694},
    "JP": {"country": "Japan", "capital": "Tokyo", "lat": 35.6762, "lng": 139.6503},
    "KR": {"country": "South Korea", "capital": "Seoul", "lat": 37.5665, "lng": 126.9780},
    "IN": {"country": "India", "capital": "New Delhi", "lat": 28.6139, "lng": 77.2090},
    "ID": {"country": "Indonesia", "capital": "Jakarta", "lat": -6.2088, "lng": 106.8456},
    "BR": {"country": "Brazil", "capital": "Brasilia", "lat": -15.7939, "lng": -47.8828},
    "MX": {"country": "Mexico", "capital": "Mexico City", "lat": 19.4326, "lng": -99.1332},
    "AR": {"country": "Argentina", "capital": "Buenos Aires", "lat": -34.6037, "lng": -58.3816},
    "CL": {"country": "Chile", "capital": "Santiago", "lat": -33.4489, "lng": -70.6693},
    "PE": {"country": "Peru", "capital": "Lima", "lat": -12.0464, "lng": -77.0428},
    "TR": {"country": "Turkey", "capital": "Ankara", "lat": 39.9334, "lng": 32.8597},
    "IR": {"country": "Iran", "capital": "Tehran", "lat": 35.6892, "lng": 51.3890},
    "SA": {"country": "Saudi Arabia", "capital": "Riyadh", "lat": 24.7136, "lng": 46.6753},
    "AE": {"country": "United Arab Emirates", "capital": "Abu Dhabi", "lat": 24.4539, "lng": 54.3773},
    "IE": {"country": "Ireland", "capital": "Dublin", "lat": 53.3498, "lng": -6.2603},
    "SE": {"country": "Sweden", "capital": "Stockholm", "lat": 59.3293, "lng": 18.0686},
    "NO": {"country": "Norway", "capital": "Oslo", "lat": 59.9139, "lng": 10.7522},
    "FI": {"country": "Finland", "capital": "Helsinki", "lat": 60.1699, "lng": 24.9384},
    "DK": {"country": "Denmark", "capital": "Copenhagen", "lat": 55.6761, "lng": 12.5683},
    "PL": {"country": "Poland", "capital": "Warsaw", "lat": 52.2297, "lng": 21.0122},
    "RS": {"country": "Serbia", "capital": "Belgrade", "lat": 44.7866, "lng": 20.4489},
    "MK": {"country": "North Macedonia", "capital": "Skopje", "lat": 41.9981, "lng": 21.4254},
    "KZ": {"country": "Kazakhstan", "capital": "Astana", "lat": 51.1694, "lng": 71.4491},
    "KG": {"country": "Kyrgyzstan", "capital": "Bishkek", "lat": 42.8746, "lng": 74.5698},
    "TJ": {"country": "Tajikistan", "capital": "Dushanbe", "lat": 38.5598, "lng": 68.7870},
    "JM": {"country": "Jamaica", "capital": "Kingston", "lat": 17.9712, "lng": -76.7936},
    "HT": {"country": "Haiti", "capital": "Port-au-Prince", "lat": 18.5944, "lng": -72.3074},
    "TT": {"country": "Trinidad and Tobago", "capital": "Port of Spain", "lat": 10.6549, "lng": -61.5019},
    "ML": {"country": "Mali", "capital": "Bamako", "lat": 12.6392, "lng": -8.0029},
    "TZ": {"country": "Tanzania", "capital": "Dodoma", "lat": -6.1630, "lng": 35.7516},
    "ZA": {"country": "South Africa", "capital": "Pretoria", "lat": -25.7479, "lng": 28.2293},
    "AU": {"country": "Australia", "capital": "Canberra", "lat": -35.2809, "lng": 149.1300},
    "NZ": {"country": "New Zealand", "capital": "Wellington", "lat": -41.2865, "lng": 174.7762},
    "DE": {"country": "Germany", "capital": "Berlin", "lat": 52.5200, "lng": 13.4050},
    "FR": {"country": "France", "capital": "Paris", "lat": 48.8566, "lng": 2.3522},
    "IT": {"country": "Italy", "capital": "Rome", "lat": 41.9028, "lng": 12.4964},
    "ES": {"country": "Spain", "capital": "Madrid", "lat": 40.4168, "lng": -3.7038},
    "CA": {"country": "Canada", "capital": "Ottawa", "lat": 45.4215, "lng": -75.6972},
}

COUNTRY_ALIASES: dict[str, str] = {
    "america": "US",
    "usa": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states of america": "US",
    "uk": "GB",
    "u.k.": "GB",
    "britain": "GB",
    "great britain": "GB",
    "england": "GB",
    "south korea": "KR",
    "republic of korea": "KR",
    "korea": "KR",
    "turkiye": "TR",
    "türkiye": "TR",
    "uae": "AE",
}
COUNTRY_NAME_TO_ISO: dict[str, str] = {
    str(value["country"]).casefold(): code for code, value in COUNTRY_CAPITALS.items()
}
COUNTRY_NAME_TO_ISO.update(COUNTRY_ALIASES)
CITY_NAME_TO_LOCATION: dict[str, dict[str, Any]] = {}
CITY_COUNTRY_TO_LOCATION: dict[str, dict[str, Any]] = {}


def _geo_lookup_key(value: Any) -> str:
    raw = str(value or "").casefold().strip()
    deaccented = "".join(
        char for char in unicodedata.normalize("NFKD", raw) if not unicodedata.combining(char)
    )
    ascii_key = re.sub(r"[^a-z0-9]+", " ", deaccented).strip()
    return ascii_key or raw


def _geo_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "data" / "country-capitals.json"


def _load_geo_country_data() -> None:
    path = _geo_data_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    countries = data.get("countries") if isinstance(data, dict) else {}
    if isinstance(countries, dict):
        for code, country in countries.items():
            iso = str(code or "").upper().strip()
            if not iso or not isinstance(country, dict):
                continue
            try:
                lat = float(country.get("lat"))
                lng = float(country.get("lng"))
            except Exception:
                continue
            country_name = str(country.get("country") or iso).strip()
            capital = str(country.get("capital") or country_name).strip()
            COUNTRY_CAPITALS[iso] = {
                "country": country_name,
                "capital": capital,
                "lat": lat,
                "lng": lng,
                "precision": str(country.get("precision") or "capital"),
            }
            for alias in [country_name, capital, iso, *(country.get("aliases") or [])]:
                key = _geo_lookup_key(alias)
                if key:
                    COUNTRY_NAME_TO_ISO.setdefault(key, iso)
                    if str(alias).strip() == capital:
                        CITY_NAME_TO_LOCATION.setdefault(
                            key,
                            {"city": capital, "country_iso": iso, "lat": lat, "lng": lng},
                        )
            translations = country.get("translations") or {}
            if isinstance(translations, dict):
                for name in translations.values():
                    key = _geo_lookup_key(name)
                    if key:
                        COUNTRY_NAME_TO_ISO.setdefault(key, iso)
    aliases = data.get("country_aliases") if isinstance(data, dict) else {}
    if isinstance(aliases, dict):
        for alias, code in aliases.items():
            iso = str(code or "").upper().strip()
            if iso in COUNTRY_CAPITALS:
                COUNTRY_NAME_TO_ISO.setdefault(str(alias), iso)
    city_aliases = data.get("city_aliases") if isinstance(data, dict) else {}
    city_country_aliases = data.get("city_country_aliases") if isinstance(data, dict) else {}
    for aliases, target in [(city_aliases, CITY_NAME_TO_LOCATION), (city_country_aliases, CITY_COUNTRY_TO_LOCATION)]:
        if not isinstance(aliases, dict):
            continue
        for alias, city in aliases.items():
            if not isinstance(city, dict):
                continue
            iso = str(city.get("country_iso") or "").upper().strip()
            if iso not in COUNTRY_CAPITALS:
                continue
            try:
                lat = float(city.get("lat"))
                lng = float(city.get("lng"))
            except Exception:
                continue
            key = str(alias or "").strip() or _geo_lookup_key(city.get("city"))
            if key:
                target[key] = {
                    "city": str(city.get("city") or COUNTRY_CAPITALS[iso]["capital"]),
                    "country_iso": iso,
                    "lat": lat,
                    "lng": lng,
                }


_load_geo_country_data()


def _country_iso(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in COUNTRY_CAPITALS:
        return upper
    return COUNTRY_NAME_TO_ISO.get(raw.casefold()) or COUNTRY_NAME_TO_ISO.get(_geo_lookup_key(raw), "")


def _city_location(value: Any, country_iso: str = "") -> dict[str, Any] | None:
    key = _geo_lookup_key(value)
    if not key:
        return None
    iso = str(country_iso or "").upper().strip()
    if iso:
        city = CITY_COUNTRY_TO_LOCATION.get(f"{key}|{iso}")
        if city:
            return city
    return CITY_NAME_TO_LOCATION.get(key)


def _country_cache_key(req: KimiTrackCountryRequest) -> str:
    import hashlib

    parts = [
        req.track_id,
        req.title,
        req.artist,
        req.album,
        req.platform_track_url,
        req.source_dataset,
        req.platform,
    ]
    raw = "\n".join(str(part or "").strip().casefold() for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _country_cache_path(storage: Storage) -> Path:
    return storage.ensure_dir("ai") / "kimi_track_country_cache.json"


def _read_country_cache(storage: Storage) -> dict[str, Any]:
    path = _country_cache_path(storage)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_country_cache(storage: Storage, cache: dict[str, Any]) -> None:
    path = _country_cache_path(storage)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _moonshot_api_base(endpoint: str) -> str:
    value = str(endpoint or "https://api.moonshot.cn/v1/chat/completions").strip().rstrip("/")
    if value.endswith("/chat/completions"):
        return value[: -len("/chat/completions")]
    return os.environ.get("KIMI_API_BASE", "https://api.moonshot.cn/v1").rstrip("/")


def _request_json(
    *,
    url: str,
    api_key: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 45.0,
) -> dict[str, Any] | list[Any]:
    import urllib.error
    import urllib.request

    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Kimi upstream HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Kimi upstream unavailable: {e.reason}") from e
    data = json.loads(raw)
    if not isinstance(data, (dict, list)):
        raise RuntimeError("Kimi upstream returned invalid JSON")
    return data


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _clean_country_result(data: dict[str, Any], *, cached: bool = False) -> dict[str, Any]:
    try:
        confidence = float(data.get("confidence") or 0)
    except Exception:
        confidence = 0.0
    iso = _country_iso(data.get("country_iso") or data.get("country_code") or data.get("country"))
    capital = COUNTRY_CAPITALS.get(iso)
    min_confidence = _env_float("ECHO_KIMI_COUNTRY_MIN_CONFIDENCE", 0.68, min_value=0.0, max_value=1.0)
    resolved = bool(data.get("resolved", True)) and bool(capital) and confidence >= min_confidence
    if not resolved:
        return {
            "ok": True,
            "resolved": False,
            "confidence": confidence,
            "cached": cached,
            "reason": str(data.get("reason") or data.get("rationale") or "country_not_confident").strip()[:500],
        }
    return {
        "ok": True,
        "resolved": True,
        "country": str(capital["country"]),
        "country_iso": iso,
        "capital": str(capital["capital"]),
        "lat": float(capital["lat"]),
        "lng": float(capital["lng"]),
        "precision": "ai_country_capital",
        "confidence": confidence,
        "cached": cached,
        "source": "kimi_web_search",
        "rationale": str(data.get("rationale") or data.get("reason") or "").strip()[:500],
        "evidence": [str(item).strip()[:300] for item in data.get("evidence", [])[:3] if str(item).strip()]
        if isinstance(data.get("evidence"), list)
        else [],
    }


def _kimi_track_country_prompt(req: KimiTrackCountryRequest) -> list[dict[str, str]]:
    user_lines = [
        f"title: {req.title}",
        f"artist: {req.artist}",
        f"album: {req.album}",
        f"city/location label: {' / '.join(part for part in [req.city, req.location] if part)}",
        f"label/genre: {req.label}",
        f"tags: {req.tags}",
        f"culture bucket: {req.culture}",
        f"source dataset: {req.source_dataset}",
        f"platform: {req.platform}",
        f"platform track url: {req.platform_track_url}",
        f"release year: {req.release_year or ''}",
    ]
    return [
        {
            "role": "system",
            "content": (
                "You identify only the most likely country associated with a music track. "
                "Use web search when the metadata is insufficient. "
                "Reliable evidence can include the artist's widely documented country, the label or scene country, "
                "or an official soundtrack/work origin when it clearly applies to this specific track. "
                "Do not infer a country merely from broad culture buckets, language guesses, genre names, "
                "or stereotypes. If evidence is weak or conflicting, return resolved=false. "
                "Your final answer must be one strict JSON object only, with keys: "
                "resolved, country, country_iso, confidence, rationale, evidence. "
                "country_iso must be an ISO 3166-1 alpha-2 code. confidence must be 0..1."
            ),
        },
        {
            "role": "user",
            "content": (
                "Find the track's country if it can be confirmed from reliable web/search evidence. "
                "Country only; do not return city-level placement. "
                "If you cannot confirm the country, return "
                '{"resolved":false,"country":"","country_iso":"","confidence":0,"rationale":"uncertain","evidence":[]}.\n\n'
                + "\n".join(user_lines)
            ),
        },
    ]


def _resolve_track_country_with_kimi(
    *,
    req: KimiTrackCountryRequest,
    api_key: str,
    endpoint: str,
    model: str,
) -> dict[str, Any]:
    timeout = float(req.timeout_seconds)
    api_base = _moonshot_api_base(endpoint)
    formula = str(os.environ.get("KIMI_WEB_SEARCH_FORMULA") or "moonshot/web-search:latest").strip()
    tools_url = f"{api_base}/formulas/{formula}/tools"
    fibers_url = f"{api_base}/formulas/{formula}/fibers"
    tools_data = _request_json(url=tools_url, api_key=api_key, timeout=timeout)
    tools = tools_data.get("tools") if isinstance(tools_data, dict) else tools_data
    if not isinstance(tools, list) or not tools:
        return {"ok": True, "resolved": False, "reason": "web_search_tools_unavailable"}

    messages: list[dict[str, Any]] = _kimi_track_country_prompt(req)
    chat_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "thinking": {"type": "disabled"},
        "max_completion_tokens": 768,
    }
    message: dict[str, Any] = {}
    for turn in range(6):
        chat_data = _request_json(url=endpoint, api_key=api_key, method="POST", payload=chat_payload, timeout=timeout)
        if not isinstance(chat_data, dict):
            return {"ok": True, "resolved": False, "reason": "invalid_chat_response"}
        message = ((chat_data.get("choices") or [{}])[0].get("message") or {})
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list) or not tool_calls:
            break
        messages.append(message)
        for call in tool_calls[:4]:
            function_call = call.get("function") or {}
            encrypted_output = ""
            try:
                fiber = _request_json(
                    url=fibers_url,
                    api_key=api_key,
                    method="POST",
                    payload=function_call,
                    timeout=timeout,
                )
                if isinstance(fiber, dict):
                    encrypted_output = str(
                        fiber.get("encrypted_output")
                        or (fiber.get("context") or {}).get("encrypted_output")
                        or ""
                    )
            except Exception as exc:
                encrypted_output = json.dumps({"error": f"web_search_failed: {exc}"}, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "name": function_call.get("name") or "web_search",
                    "content": encrypted_output or '{"error":"empty_web_search_result"}',
                }
            )
        for call in tool_calls[4:]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "name": ((call.get("function") or {}).get("name") or "web_search"),
                    "content": '{"error":"tool_call_limit"}',
                }
            )
        chat_payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "thinking": {"type": "disabled"},
            "max_completion_tokens": 768,
        }
    else:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Stop searching now. Based only on the evidence already gathered in this conversation, "
                    "return the final strict JSON object. If the gathered evidence is still insufficient, "
                    'return {"resolved":false,"country":"","country_iso":"","confidence":0,'
                    '"rationale":"uncertain","evidence":[]}.'
                ),
            }
        )
        final_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "max_completion_tokens": 768,
        }
        chat_data = _request_json(url=endpoint, api_key=api_key, method="POST", payload=final_payload, timeout=timeout)
        if not isinstance(chat_data, dict):
            return {"ok": True, "resolved": False, "reason": "web_search_round_limit"}
        message = ((chat_data.get("choices") or [{}])[0].get("message") or {})
    content = str(message.get("content") or "").strip()
    if not content:
        return {"ok": True, "resolved": False, "reason": "empty_country_response"}
    return _clean_country_result(_extract_json_object(content), cached=False)


def _mainline_catalog(storage: Storage):
    return get_lightweight_catalog(storage, metadata_rel=_mainline_metadata_setting())


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
    return str(os.environ.get("ECHO_COOKIE_SECURE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _mainline_local_recommender_enabled() -> bool:
    return _env_bool("ECHO_MAINLINE_LOCAL_RECOMMENDER_ENABLED", True)


def _raise_mainline_worker_required() -> None:
    raise HTTPException(
        status_code=503,
        detail=(
            "mainline worker is not configured. Set ECHO_MAINLINE_WORKER_URL "
            "and ECHO_MAINLINE_WORKER_TOKEN, or enable the local recommender "
            "with ECHO_MAINLINE_LOCAL_RECOMMENDER_ENABLED=true and install "
            "the full research dependencies."
        ),
    )


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
    expected = str(
        os.environ.get("ECHO_WORKER_SHARED_TOKEN", "") or os.environ.get("ECHO_MAINLINE_WORKER_TOKEN", "")
    ).strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="worker token enforcement is enabled but no token is configured",
        )
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


def _hydrate_worker_mainline_metadata(data: dict[str, Any], storage: Storage) -> dict[str, Any]:
    try:
        catalog = _mainline_catalog(storage)
    except Exception:
        return data

    def hydrate_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        track_id = str(item.get("track_id") or item.get("trackId") or item.get("id") or "").strip()
        if not track_id:
            return
        try:
            metadata = catalog.track(track_id)
        except Exception:
            return
        for field in ("album_id", "release_date", "release_year", "year", "era"):
            value = str(metadata.get(field) or "").strip()
            if value and not str(item.get(field) or "").strip():
                item[field] = value

    for key in ("seeds", "recommendations", "items"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                hydrate_item(item)
        else:
            hydrate_item(value)
    hydrate_item(data.get("track"))
    return data


def _validate_upload_audio(filename: str, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in SUPPORTED_UPLOAD_AUDIO_EXTENSIONS:
        return suffix
    if str(content_type or "").lower().startswith("audio/"):
        return suffix or ".audio"
    allowed = ", ".join(sorted(SUPPORTED_UPLOAD_AUDIO_EXTENSIONS))
    raise HTTPException(
        status_code=415,
        detail=f"unsupported audio format. Supported extensions: {allowed}",
    )


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
    cmd.extend(
        [
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate_hz),
            "-b:a",
            str(bitrate),
            str(target),
        ]
    )
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
            _set_audio_tag(tags, key, value)
    return tags


def _tag_key_variants(key: Any) -> list[str]:
    raw = str(key or "").strip().lower()
    if not raw:
        return []
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    return list(dict.fromkeys(item for item in (raw, normalized, compact) if item))


def _decode_tag_bytes(value: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "utf-16-be", "utf-16-le", "latin-1"):
        try:
            text = value.decode(encoding).strip("\x00").strip()
        except Exception:
            continue
        if text:
            return text
    return ""


def _tag_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return _decode_tag_bytes(value)
    if isinstance(value, (list, tuple, set)):
        return "; ".join(text for item in value if (text := _tag_text(item)))
    text_values = getattr(value, "text", None)
    if isinstance(text_values, (list, tuple)):
        return "; ".join(str(item).strip() for item in text_values if str(item).strip())
    data = getattr(value, "data", None)
    if isinstance(data, bytes):
        return _decode_tag_bytes(data)
    return str(value).strip()


def _set_audio_tag(tags: dict[str, str], key: Any, value: Any) -> None:
    text = _tag_text(value)
    if not text:
        return
    for variant in _tag_key_variants(key):
        tags.setdefault(variant, text)
    normalized_key = str(key or "").strip().lower()
    if normalized_key.startswith("txxx:"):
        for variant in _tag_key_variants(normalized_key.split(":", 1)[1]):
            tags.setdefault(variant, text)
    if normalized_key.startswith("----:"):
        for variant in _tag_key_variants(normalized_key.rsplit(":", 1)[-1]):
            tags.setdefault(variant, text)


MUTAGEN_TAG_ALIASES = {
    "tit2": "title",
    "tpe1": "artist",
    "tpe2": "album_artist",
    "tcom": "composer",
    "talb": "album",
    "tcon": "genre",
    "tdrc": "date",
    "tyer": "year",
    "comm": "comment",
    "\xa9nam": "title",
    "\xa9art": "artist",
    "aart": "album_artist",
    "\xa9alb": "album",
    "\xa9gen": "genre",
    "\xa9day": "date",
    "\xa9cmt": "comment",
    "desc": "description",
    "ldes": "description",
}


def _probe_audio_tags_mutagen(path: Path) -> dict[str, str]:
    try:
        from mutagen import File as MutagenFile
    except Exception:
        return {}
    tags: dict[str, str] = {}
    try:
        audio = MutagenFile(str(path), easy=False)
    except Exception:
        audio = None
    raw_tags = getattr(audio, "tags", None)
    if raw_tags:
        _collect_mutagen_tags(raw_tags, tags)
    if not tags and path.suffix.lower() in {".mp3", ".aif", ".aiff"}:
        try:
            from mutagen.id3 import ID3

            _collect_mutagen_tags(ID3(str(path)), tags)
        except Exception:
            pass
    return tags


def _collect_mutagen_tags(raw_tags: Any, tags: dict[str, str]) -> None:
    for key, value in raw_tags.items():
        _set_audio_tag(tags, key, value)
        normalized_key = str(key or "").strip().lower()
        alias = MUTAGEN_TAG_ALIASES.get(normalized_key)
        if alias:
            _set_audio_tag(tags, alias, value)
        desc = str(getattr(value, "desc", "") or "").strip()
        if desc:
            _set_audio_tag(tags, desc, value)
        if normalized_key.startswith("txxx:"):
            _set_audio_tag(tags, normalized_key.split(":", 1)[1], value)
        if normalized_key.startswith("----:"):
            _set_audio_tag(tags, normalized_key.rsplit(":", 1)[-1], value)


def _probe_audio_tags_ffprobe(path: Path) -> dict[str, str]:
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


def _probe_audio_tags(path: Path) -> dict[str, str]:
    tags = _probe_audio_tags_mutagen(path)
    for key, value in _probe_audio_tags_ffprobe(path).items():
        tags.setdefault(key, value)
    return tags


def _tag_first(tags: dict[str, str], *names: str) -> str:
    for name in names:
        for key in _tag_key_variants(name):
            value = str(tags.get(key) or "").strip()
            if value:
                return value
    return ""


def _split_location_label(value: Any) -> tuple[str, str]:
    text = " ".join(str(value or "").replace("\n", ",").split())
    if not text:
        return "", ""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return "", ""


def _parse_coordinate_scalar(value: Any, *, min_value: float, max_value: float) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    sign = -1.0 if re.search(r"[sw]\s*$", text, flags=re.IGNORECASE) else 1.0
    cleaned = re.sub(r"[nsew]", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("+", "").strip()
    try:
        out = float(cleaned)
    except Exception:
        return None
    out *= sign
    if min_value <= out <= max_value:
        return out
    return None


def _parse_coordinate_pair(value: Any) -> tuple[float | None, float | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    iso_match = re.match(r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?/?\s*$", text)
    if iso_match:
        lat = _parse_coordinate_scalar(iso_match.group(1), min_value=-90.0, max_value=90.0)
        lng = _parse_coordinate_scalar(iso_match.group(2), min_value=-180.0, max_value=180.0)
        return lat, lng
    pair_match = re.search(
        r"([+-]?\d{1,2}(?:\.\d+)?)\s*[,;/ ]+\s*([+-]?\d{1,3}(?:\.\d+)?)",
        text,
    )
    if not pair_match:
        return None, None
    lat = _parse_coordinate_scalar(pair_match.group(1), min_value=-90.0, max_value=90.0)
    lng = _parse_coordinate_scalar(pair_match.group(2), min_value=-180.0, max_value=180.0)
    return lat, lng


def _coordinates_from_tags(tags: dict[str, str]) -> tuple[float | None, float | None]:
    lat = _parse_coordinate_scalar(
        _tag_first(tags, "lat", "latitude", "geo_latitude", "geo:lat", "gps_latitude", "gpslatitude"),
        min_value=-90.0,
        max_value=90.0,
    )
    lng = _parse_coordinate_scalar(
        _tag_first(
            tags,
            "lng",
            "lon",
            "long",
            "longitude",
            "geo_longitude",
            "geo:lon",
            "gps_longitude",
            "gpslongitude",
        ),
        min_value=-180.0,
        max_value=180.0,
    )
    if lat is not None and lng is not None:
        return lat, lng
    for key in (
        "coordinates",
        "coordinate",
        "gps",
        "geo",
        "geolocation",
        "geo_location",
        "location_coordinates",
        "com.apple.quicktime.location.iso6709",
        "iso6709",
    ):
        parsed_lat, parsed_lng = _parse_coordinate_pair(_tag_first(tags, key))
        if parsed_lat is not None and parsed_lng is not None:
            return parsed_lat, parsed_lng
    return None, None


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
    location = _tag_first(
        tags,
        "location",
        "artist_location",
        "artist location",
        "com.apple.iTunes.location",
        "com.apple.quicktime.location.name",
        "venue",
        "place",
    )
    city = _tag_first(tags, "city", "location_city", "artist_city", "venue")
    country = _tag_first(
        tags,
        "country",
        "country_name",
        "location_country",
        "artist_country",
        "com.apple.iTunes.country",
        "musicbrainz_albumartistcountry",
    )
    country_code = _tag_first(tags, "country_code", "countrycode", "iso_country", "isrc_country")
    parsed_city, parsed_country = _split_location_label(location)
    country = country or parsed_country
    city = city or parsed_city
    lat, lng = _coordinates_from_tags(tags)
    iso = _country_iso(country_code or country)
    capital = COUNTRY_CAPITALS.get(iso)
    city_location = _city_location(city or location, iso)
    if lat is not None and lng is not None:
        return {
            "country": capital["country"] if capital else (country or country_code or "embedded_location"),
            "country_code": iso or country_code,
            "city": city or (capital["capital"] if capital else country or location or "embedded_location"),
            "location": location,
            "lat": lat,
            "lng": lng,
            "location_precision": "audio_tag_coordinates",
            "location_note": "From embedded audio file metadata",
        }
    if city_location:
        city_iso = str(city_location["country_iso"])
        city_country = COUNTRY_CAPITALS.get(city_iso, {})
        return {
            "country": city_country.get("country", city_iso),
            "country_code": city_iso,
            "city": city_location.get("city") or city or location,
            "location": location,
            "lat": float(city_location["lat"]),
            "lng": float(city_location["lng"]),
            "location_precision": "audio_tag_city",
            "location_note": "From embedded audio city metadata",
        }
    if capital:
        return {
            "country": capital["country"],
            "country_code": iso,
            "city": city or capital["capital"],
            "location": location,
            "lat": float(capital["lat"]),
            "lng": float(capital["lng"]),
            "location_precision": "audio_tag_country_capital",
            "location_note": "From embedded audio country metadata",
        }
    if country or country_code or city or location or (lat is not None and lng is not None):
        return {
            "country": country or country_code or "mystery_place",
            "country_code": country_code,
            "city": city or location or "central_pacific",
            "location": location,
            "lat": 0.0,
            "lng": -160.0,
            "location_precision": "audio_tag_unresolved",
            "location_note": "Embedded audio metadata has a place label but no confidently resolved country",
        }
    return {
        "country": "mystery_place",
        "city": "central_pacific",
        "lat": 0.0,
        "lng": -160.0,
        "location_precision": "pacific_default",
        "location_note": "Upload audio has no embedded region, country, city, or coordinate tags",
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
                return JSONResponse(
                    status_code=404,
                    content={"detail": "worker mode only exposes mainline API"},
                )
            expected = str(
                os.environ.get("ECHO_WORKER_SHARED_TOKEN", "") or os.environ.get("ECHO_MAINLINE_WORKER_TOKEN", "")
            ).strip()
            actual = str(request.headers.get("X-Echo-Worker-Token", "")).strip()
            if not expected:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "worker token enforcement is enabled but no token is configured"},
                )
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
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Echo-Worker-Token"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

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
            catalog = _mainline_catalog(storage)
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
        return {
            "ok": True,
            "session_id": session_id,
            "count": len(items),
            "seeded": seeded,
            "items": items,
        }

    @app.post("/api/favorites")
    def api_add_favorite(payload: dict[str, Any], request: Request, response: Response):
        session_id = _session_id_from_request(request, response, favorite_store)
        track = payload.get("track") if isinstance(payload.get("track"), dict) else payload
        try:
            item = favorite_store.upsert_favorite(session_id, track)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "ok": True,
            "session_id": session_id,
            "item": item,
            "items": favorite_store.list_favorites(session_id),
        }

    @app.delete("/api/favorites")
    def api_remove_favorite_by_query(track_key: str, request: Request, response: Response):
        session_id = _session_id_from_request(request, response, favorite_store)
        removed = favorite_store.remove_favorite(session_id, track_key)
        return {
            "ok": True,
            "session_id": session_id,
            "removed": removed,
            "items": favorite_store.list_favorites(session_id),
        }

    @app.delete("/api/favorites/{track_key:path}")
    def api_remove_favorite(track_key: str, request: Request, response: Response):
        session_id = _session_id_from_request(request, response, favorite_store)
        removed = favorite_store.remove_favorite(session_id, track_key)
        return {
            "ok": True,
            "session_id": session_id,
            "removed": removed,
            "items": favorite_store.list_favorites(session_id),
        }

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
                data["worker"] = {
                    "configured": True,
                    "online": True,
                    "url": _mainline_worker_url(),
                }
                return data
            except HTTPException as e:
                try:
                    data = _mainline_catalog(storage).status()
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
            data = _mainline_catalog(storage).status()
            data["worker"] = {"configured": False, "online": False, "url": ""}
            return data
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/cultures")
    def api_mainline_cultures(prefer_cuda: bool = False):
        try:
            return _mainline_catalog(storage).cultures()
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
            catalog = _mainline_catalog(storage)
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
            catalog = _mainline_catalog(storage)
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
            return _mainline_catalog(storage).track(track_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/mainline/audio/{track_id}")
    def api_mainline_audio(track_id: str, prefer_cuda: bool = False):
        try:
            catalog = _mainline_catalog(storage)
            path, media_type = catalog.audio_file(track_id)
            return FileResponse(str(path), media_type=media_type, headers={"Accept-Ranges": "bytes"})
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except FileNotFoundError:
            try:
                track = _mainline_catalog(storage).track(track_id)
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
            return _hydrate_worker_mainline_metadata(
                _proxy_worker_json("/api/mainline/recommend", req.dict()),
                storage,
            )
        if not _mainline_local_recommender_enabled():
            _raise_mainline_worker_required()
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
            platform = get_mainline_platform(storage, prefer_cuda=req.prefer_cuda, **_mainline_platform_paths())
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
        upload_provider = _upload_embedding_provider()
        worker_configured = bool(_mainline_worker_url())
        local_recommender_enabled = _mainline_local_recommender_enabled()
        return {
            "ok": True,
            "mode": (
                "gemini"
                if upload_provider == "gemini"
                else "worker"
                if worker_configured
                else "local"
                if local_recommender_enabled
                else "unavailable"
            ),
            "worker_configured": worker_configured,
            "local_recommender_enabled": local_recommender_enabled,
            "upload_embedding": {
                "provider": upload_provider,
                "worker_bypass": upload_provider == "gemini",
                "mainline_tracks": _mainline_tracks_setting(),
                "mainline_model": _mainline_model_setting(),
                "gemini": _public_gemini_embedding_settings(),
            },
            "extensions": sorted(SUPPORTED_UPLOAD_AUDIO_EXTENSIONS),
            "accept": UPLOAD_ACCEPT_ATTRIBUTE,
            "max_upload_mb": 200,
            "culturemert": {
                "model_id": _culturemert_runtime_settings()["culturemert_model_id"],
                "cache_dir": _culturemert_runtime_settings()["culturemert_cache_dir"] or "",
                "revision": _culturemert_runtime_settings()["culturemert_revision"] or "",
                "local_files_only": bool(_culturemert_runtime_settings()["culturemert_local_files_only"]),
            },
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
        upload_embedding_provider = _upload_embedding_provider()
        if upload_embedding_provider not in {"culturemert", "gemini"}:
            raise HTTPException(
                status_code=500,
                detail=f"unsupported ECHO_UPLOAD_EMBEDDING_PROVIDER={upload_embedding_provider}; use culturemert or gemini",
            )
        if _mainline_worker_url() and upload_embedding_provider != "gemini":
            return _hydrate_worker_mainline_metadata(
                _proxy_worker_upload_recommend(
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
                ),
                storage,
            )
        if not _mainline_local_recommender_enabled():
            _raise_mainline_worker_required()
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
            "artist": artist
            or _tag_first(tag_info, "artist", "album_artist", "albumartist", "composer")
            or "Uploaded audio",
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
            platform = get_mainline_platform(storage, prefer_cuda=prefer_cuda, **_mainline_platform_paths())
            if upload_embedding_provider == "gemini":
                emb, embedding_meta = _embed_uploaded_audio_with_gemini(
                    audio_path=analysis_path,
                    title=str(upload_info.get("title") or title or filename),
                    max_seconds=max_seconds,
                    window_count=window_count,
                    window_strategy=window_strategy,
                    window_aggregate=window_aggregate,
                )
                result = platform.recommend_embedding(
                    embedding=emb,
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
                )
                result["embedding"] = embedding_meta
                result.setdefault("algorithm", {})
                result["algorithm"]["backbone"] = "Gemini Embedding 2 audio embeddings"
                result["algorithm"]["reranker"] = (
                    "uploaded Gemini audio seed -> DCAS latent encoding -> OT relevance + calibrated cultural reranking"
                )
                result["warnings"] = list(result.get("warnings") or []) + _warn_if_mainline_not_gemini(platform)
            else:
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
                    **_culturemert_runtime_settings(),
                )
            return result
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            if upload_embedding_provider == "gemini" and _looks_like_gemini_embedding_error(e):
                settings = _public_gemini_embedding_settings()
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "gemini_embedding_unavailable",
                        "message": str(e)[:2000],
                        "model_id": settings["model_id"],
                        "api_base": settings["api_base"],
                        "has_api_key": bool(settings["has_api_key"]),
                        "output_dimensionality": int(settings["output_dimensionality"]),
                        "hint": (
                            "Set GEMINI_API_KEY and ECHO_UPLOAD_EMBEDDING_PROVIDER=gemini on the web server. "
                            "For best recommendation quality, also point ECHO_MAINLINE_TRACKS_PATH and ECHO_MAINLINE_MODEL_PATH "
                            "to Gemini-built artifacts."
                        ),
                    },
                )
            if _looks_like_culturemert_load_error(e):
                settings = _culturemert_runtime_settings()
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "culturemert_model_unavailable",
                        "message": str(e)[:2000],
                        "model_id": settings["culturemert_model_id"],
                        "cache_dir": settings["culturemert_cache_dir"] or "",
                        "revision": settings["culturemert_revision"] or "",
                        "local_files_only": bool(settings["culturemert_local_files_only"]),
                        "hint": (
                            "Run scripts/preload_culturemert_model.py on the local worker machine, "
                            "or set ECHO_CULTUREMERT_LOCAL_FILES_ONLY=true after the model is cached."
                        ),
                    },
                )
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

    @app.post("/api/ai/kimi/track-country")
    def api_kimi_track_country(req: KimiTrackCountryRequest):
        if not (str(req.title or "").strip() or str(req.artist or "").strip() or str(req.platform_track_url or "").strip()):
            return {"ok": True, "resolved": False, "reason": "insufficient_track_metadata"}
        local = _load_local_kimi_config()
        api_key = str(req.api_key or "").strip() or str(local.get("api_key") or "").strip()
        if not api_key:
            return {"ok": True, "resolved": False, "reason": "kimi_api_key_not_configured"}
        endpoint = (req.endpoint.strip() if req.endpoint else "") or local.get(
            "endpoint", "https://api.moonshot.cn/v1/chat/completions"
        )
        if not endpoint.startswith("https://"):
            raise HTTPException(status_code=400, detail="Kimi endpoint must use https")
        model = (req.model.strip() if req.model else "") or local.get("model", "kimi-k2.6")
        key = _country_cache_key(req)
        cache = _read_country_cache(storage)
        cached = cache.get(key)
        if isinstance(cached, dict):
            return _clean_country_result(cached, cached=True)
        try:
            result = _resolve_track_country_with_kimi(req=req, api_key=api_key, endpoint=endpoint, model=model)
        except Exception as e:
            return {"ok": True, "resolved": False, "reason": str(e)[:1000]}
        cache[key] = {
            "resolved": bool(result.get("resolved")),
            "country": result.get("country", ""),
            "country_iso": result.get("country_iso", ""),
            "confidence": float(result.get("confidence") or 0),
            "rationale": result.get("rationale") or result.get("reason") or "",
            "evidence": result.get("evidence") or [],
            "updated_at": time.time(),
        }
        _write_country_cache(storage, cache)
        return result

    @app.post("/api/ai/kimi/chat")
    def api_kimi_chat(req: KimiChatRequest):
        local = _load_local_kimi_config()
        api_key = str(req.api_key or "").strip() or str(local.get("api_key") or "").strip()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Kimi API key is not configured. Set it in settings.html, KIMI_API_KEY, or configs/secrets/kimi.local.json.",
            )
        endpoint = (req.endpoint.strip() if req.endpoint else "") or local.get(
            "endpoint", "https://api.moonshot.cn/v1/chat/completions"
        )
        if not endpoint.startswith("https://"):
            raise HTTPException(status_code=400, detail="Kimi endpoint must use https")
        model = (req.model.strip() if req.model else "") or local.get("model", "kimi-k2.6")
        payload = _kimi_request_payload(req, model)
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
            raise HTTPException(
                status_code=504,
                detail=f"Kimi upstream timed out after {float(req.timeout_seconds):.0f}s",
            )
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
        reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
        return {
            "ok": True,
            "content": content,
            "reasoning_content": reasoning_content,
            "thinking_mode": _normalize_kimi_thinking_mode(req.thinking_mode),
            "raw": data,
        }

    @app.post("/api/ai/kimi/chat/stream")
    def api_kimi_chat_stream(req: KimiChatRequest):
        local = _load_local_kimi_config()
        api_key = str(req.api_key or "").strip() or str(local.get("api_key") or "").strip()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Kimi API key is not configured. Set it in settings.html, KIMI_API_KEY, or configs/secrets/kimi.local.json.",
            )
        endpoint = (req.endpoint.strip() if req.endpoint else "") or local.get(
            "endpoint", "https://api.moonshot.cn/v1/chat/completions"
        )
        if not endpoint.startswith("https://"):
            raise HTTPException(status_code=400, detail="Kimi endpoint must use https")
        model = (req.model.strip() if req.model else "") or local.get("model", "kimi-k2.6")
        body = json.dumps(_kimi_request_payload(req, model, stream=True), ensure_ascii=False).encode("utf-8")

        def events():
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
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith("event:"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_text = line[5:].strip()
                        if not data_text:
                            continue
                        if data_text == "[DONE]":
                            yield _sse_event("done", {"ok": True, "finish_reason": "stop"})
                            return
                        try:
                            chunk = json.loads(data_text)
                        except json.JSONDecodeError:
                            continue
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        reasoning_delta = _string_delta(
                            delta.get("reasoning_content") or delta.get("reasoning") or delta.get("reasoningContent")
                        )
                        content_delta = _string_delta(delta.get("content"))
                        if reasoning_delta:
                            yield _sse_event("reasoning", {"delta": reasoning_delta})
                        if content_delta:
                            yield _sse_event("content", {"delta": content_delta})
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            yield _sse_event("done", {"ok": True, "finish_reason": finish_reason})
                            return
                yield _sse_event("done", {"ok": True, "finish_reason": "stop"})
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:4000]
                yield _sse_event("error", {"status": e.code, "detail": detail})
            except urllib.error.URLError as e:
                yield _sse_event("error", {"status": 502, "detail": str(e.reason)})
            except TimeoutError:
                yield _sse_event(
                    "error",
                    {"status": 504, "detail": f"Kimi upstream timed out after {float(req.timeout_seconds):.0f}s"},
                )
            except Exception as e:
                yield _sse_event("error", {"status": 500, "detail": str(e)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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
        tasks_path = storage.resolve_rel("pal/v4_main_annotation/pal_tasks.json")
        if not tasks_path.exists():
            raise HTTPException(status_code=404, detail="PAL tasks not found. Run pal_tasks first.")
        with open(tasks_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    @app.get("/api/pal/audio")
    def stream_audio(path: str):
        """Stream an audio file by path (resolved relative to storage root)."""
        try:
            p = storage.resolve_rel(path)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid path")
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="audio not found")
        ext = p.suffix.lower()
        media_type = "audio/mpeg" if ext == ".mp3" else "audio/wav" if ext == ".wav" else "audio/octet-stream"
        return FileResponse(str(p), media_type=media_type, headers={"Accept-Ranges": "bytes"})

    @app.post("/api/pal/annotate")
    def save_annotation(ann: dict):
        """Save a single annotation to JSONL file."""
        out_dir = storage.ensure_dir("pal/v4_main_annotation")
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
        out_file = storage.resolve_rel("pal/v4_main_annotation/annotations.jsonl")
        if not out_file.exists():
            return {"count": 0, "total": 60}
        with open(out_file, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        return {"count": count, "total": 60}

    @app.post("/api/pal/export")
    def export_annotations():
        """Export annotations as CSV for constraint building."""
        out_file = storage.resolve_rel("pal/v4_main_annotation/annotations.jsonl")
        csv_file = storage.resolve_rel("pal/v4_main_annotation/annotated.csv")
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
        tasks_json = storage.resolve_rel("pal/v4_main_annotation/pal_tasks.json")
        if tasks_json.exists():
            with open(tasks_json, "r", encoding="utf-8") as f:
                for t in json.load(f):
                    tasks_map[(t["track_id_a"], t["track_id_b"])] = t

        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(
                [
                    "task_id",
                    "track_id_a",
                    "track_id_b",
                    "judgment",
                    "similar",
                    "rationale",
                    "annotator",
                ]
            )
            for a in anns:
                judgment = a.get("judgment", "")
                similar = "yes" if judgment in ("a", "b") else ("no" if judgment == "neither" else "")
                writer.writerow(
                    [
                        a["task_id"],
                        a["track_id_a"],
                        a["track_id_b"],
                        judgment,
                        similar,
                        a.get("rationale", ""),
                        a.get("annotator", ""),
                    ]
                )
        return {"csv": str(csv_file), "count": len(anns)}

    web_dir = Path("web")
    dist = Path("web/dist")
    prototype_dir = Path("web_prototype")
    if prototype_dir.exists():
        app.mount(
            "/prototype",
            StaticFiles(directory=str(prototype_dir), html=True),
            name="prototype",
        )
    # PAL annotation UI
    pal_html = dist / "pal.html"
    if pal_html.exists():
        app.mount("/pal", StaticFiles(directory=str(pal_html.parent), html=False), name="pal")
    if (web_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    elif (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
    elif prototype_dir.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(prototype_dir), html=True),
            name="prototype-root",
        )
    else:
        dist.mkdir(parents=True, exist_ok=True)
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web-empty")

    return app


app = create_app()
