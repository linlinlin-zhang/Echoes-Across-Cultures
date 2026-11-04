"""Read-only cloud catalog and existing compressed audio, independent of the tunnel."""

from __future__ import annotations
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from dcas_server.lightweight_catalog import get_lightweight_catalog
from dcas_server.paths import Storage

root = Path(__file__).resolve().parent
audio_root = Path(os.environ.get("ECHO_EDGE_AUDIO_ROOT", "/srv/echo/cloud_audio_10gb")).resolve()
catalog = get_lightweight_catalog(Storage(root / "storage"))
audio = json.loads((root / "audio-index.json").read_text(encoding="utf-8"))
# Warm the shared dictionaries before accepting simultaneous visitors.
catalog.track(next(iter(catalog.by_id)))
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def edge_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Echo-Edge"] = "cloud"
    return response


@app.exception_handler(KeyError)
async def missing(request, exc):
    return JSONResponse({"detail": "Track not found"}, status_code=404)


@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse({"detail": "Invalid catalog query"}, status_code=400)


@app.get("/api/edge/health")
def health():
    return {
        "ok": True,
        "mode": "cloud_read_only",
        "catalog_tracks": len(catalog.by_id),
        "cloud_audio_tracks": len(audio),
    }


@app.get("/api/mainline/status")
def status():
    data = catalog.status()
    data.pop("metadata_path", None)
    data["worker"] = {"configured": False, "online": False, "url": ""}
    data["delivery"] = "cloud_catalog"
    return data


@app.get("/api/mainline/cultures")
def cultures():
    return catalog.cultures()


@app.get("/api/mainline/catalog")
def listing(
    culture: str | None = None,
    source_dataset: str | None = None,
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=24, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=30000),
    random_seed: int | None = 42,
    exclude_low_signal: bool = True,
):
    return catalog.catalog(
        culture=culture,
        source_dataset=source_dataset,
        q=q,
        limit=limit,
        offset=offset,
        random_seed=random_seed,
        exclude_low_signal=exclude_low_signal,
    )


@app.get("/api/mainline/random")
def random_track(
    culture: str | None = None,
    source_dataset: str | None = None,
    random_seed: int | None = 42,
    exclude_low_signal: bool = True,
):
    return catalog.random_track(
        culture=culture, source_dataset=source_dataset, random_seed=random_seed, exclude_low_signal=exclude_low_signal
    )


@app.get("/api/mainline/tracks/{track_id}")
def track(track_id: str):
    return catalog.track(track_id)


@app.get("/api/mainline/audio/{track_id}")
def track_audio(track_id: str):
    if track_id not in catalog.by_id:
        raise HTTPException(404, "Track not found")
    relative = audio.get(track_id)
    if not relative:
        raise HTTPException(503, "Cloud audio unavailable")
    file = (audio_root / relative).resolve()
    if not file.is_relative_to(audio_root) or not file.is_file():
        raise HTTPException(503, "Cloud audio unavailable")
    return Response(
        headers={
            "X-Accel-Redirect": "/__echo_audio/" + file.relative_to(audio_root).as_posix(),
            "Content-Type": "audio/mp4",
        }
    )
