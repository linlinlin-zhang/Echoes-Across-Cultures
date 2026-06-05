from __future__ import annotations

import csv
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .paths import Storage


DEFAULT_METADATA_REL = "public/merged/metadata_merged.csv"
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


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm_key(value: Any) -> str:
    text = _clean(value).lower()
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _first_clean(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _clean(row.get(key))
        if value:
            return value
    return ""


def _release_year_from_text(value: Any) -> str:
    match = re.search(r"(?:19|20)\d{2}", _clean(value))
    return match.group(0) if match else ""


def _release_year(row: dict[str, Any], release_date: str = "") -> str:
    return _first_clean(row, ("release_year", "year")) or _release_year_from_text(release_date)


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


def _storage_or_absolute_path(storage: Storage, value: str) -> Path:
    path = Path(_clean(value))
    if path.is_absolute():
        return path.resolve()
    return storage.resolve_rel(value)


class LightweightMainlineCatalog:
    """Metadata-only catalog used by cloud deployments that do not run models."""

    def __init__(self, storage: Storage, *, metadata_rel: str = DEFAULT_METADATA_REL) -> None:
        self.storage = storage
        self.metadata_path = _storage_or_absolute_path(storage, metadata_rel)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"mainline metadata missing: {self.metadata_path}")
        self.loaded_at = time.time()
        self.metadata_mtime_ns = self.metadata_path.stat().st_mtime_ns
        self.rows = self._load_rows()
        self.by_id = {str(row.get("track_id")): row for row in self.rows if _clean(row.get("track_id"))}
        self.culture_counts = Counter(_clean(row.get("culture")) for row in self.rows if _clean(row.get("culture")))
        self.source_counts = Counter(
            _clean(row.get("source_dataset")) for row in self.rows if _clean(row.get("source_dataset"))
        )

    def _refresh_metadata_if_needed(self) -> None:
        mtime_ns = self.metadata_path.stat().st_mtime_ns
        if mtime_ns == self.metadata_mtime_ns:
            return
        self.loaded_at = time.time()
        self.metadata_mtime_ns = mtime_ns
        self.rows = self._load_rows()
        self.by_id = {str(row.get("track_id")): row for row in self.rows if _clean(row.get("track_id"))}
        self.culture_counts = Counter(_clean(row.get("culture")) for row in self.rows if _clean(row.get("culture")))
        self.source_counts = Counter(
            _clean(row.get("source_dataset")) for row in self.rows if _clean(row.get("source_dataset"))
        )

    def status(self) -> dict[str, Any]:
        self._refresh_metadata_if_needed()
        description_count = sum(1 for row in self.rows if _clean(row.get("description")))
        album_description_count = sum(1 for row in self.rows if _clean(row.get("album_description")))
        tag_count = sum(1 for row in self.rows if _clean(row.get("tags")))
        description_sources = Counter(
            _clean(row.get("description_source")) or "unknown" for row in self.rows if _clean(row.get("description"))
        )
        return {
            "ok": True,
            "mode": "lightweight_catalog",
            "loaded_at": self.loaded_at,
            "metadata_path": str(self.metadata_path),
            "n_tracks": int(len(self.rows)),
            "n_metadata_rows": int(len(self.rows)),
            "metadata_coverage": {
                "description_rows": int(description_count),
                "missing_description_rows": int(max(0, len(self.rows) - description_count)),
                "album_description_rows": int(album_description_count),
                "tagged_rows": int(tag_count),
                "description_ratio": float(description_count / max(1, len(self.rows))),
                "description_sources": dict(sorted(description_sources.items())),
            },
            "cultures": dict(sorted(self.culture_counts.items())),
            "sources": dict(sorted(self.source_counts.items())),
        }

    def cultures(self) -> dict[str, Any]:
        self._refresh_metadata_if_needed()
        return {
            "ok": True,
            "mode": "lightweight_catalog",
            "cultures": [{"culture": name, "count": int(count)} for name, count in sorted(self.culture_counts.items())],
            "sources": [
                {"source_dataset": name, "count": int(count)} for name, count in sorted(self.source_counts.items())
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
        self._refresh_metadata_if_needed()
        limit = max(1, min(200, int(limit)))
        culture_key = _clean(culture)
        source_key = _clean(source_dataset)
        query = _norm_key(q)
        query_terms = [term for term in query.split(" ") if term]

        keep: list[dict[str, Any]] = []
        for row in self.rows:
            if culture_key and _clean(row.get("culture")) != culture_key:
                continue
            if source_key and _clean(row.get("source_dataset")) != source_key:
                continue
            if exclude_low_signal and self._is_low_signal(row):
                continue
            if query_terms and not self._matches_query(row, query_terms):
                continue
            keep.append(row)

        if random_seed is not None:
            random.Random(int(random_seed)).shuffle(keep)

        return {
            "ok": True,
            "mode": "lightweight_catalog",
            "request": {
                "culture": culture_key,
                "source_dataset": source_key,
                "q": query,
                "limit": limit,
                "random_seed": random_seed,
                "exclude_low_signal": bool(exclude_low_signal),
            },
            "total_available": int(len(keep)),
            "items": [self._track_payload(row) for row in keep[:limit]],
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
        return {
            "ok": True,
            "mode": "lightweight_catalog",
            "track": items[0],
            "request": result["request"],
            "total_available": result["total_available"],
        }

    def track(self, track_id: str) -> dict[str, Any]:
        self._refresh_metadata_if_needed()
        row = self.by_id.get(str(track_id))
        if row is None:
            raise KeyError(f"track not found: {track_id}")
        return self._track_payload(row)

    def audio_file(self, track_id: str) -> tuple[Path, str]:
        self._refresh_metadata_if_needed()
        row = self.by_id.get(str(track_id))
        if row is None:
            raise KeyError(f"track not found: {track_id}")
        path = _metadata_audio_path(_clean(row.get("audio_path")), self.metadata_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"audio not found: {path}")
        return path, _media_type(path)

    def _load_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if _clean(row.get("track_id")):
                    rows.append(dict(row))
        return rows

    def _track_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        track_id = _clean(row.get("track_id"))
        release_date = _first_clean(
            row,
            (
                "release_date",
                "releaseDate",
                "releaseDateTime",
                "album_release_date",
                "collection_release_date",
                "date",
                "fma_track_date_recorded",
            ),
        )
        release_year = _release_year(row, release_date)
        return {
            "track_id": track_id,
            "rank": None,
            "title": _first_clean(row, ("title", "name", "track_name")) or track_id,
            "artist": _first_clean(row, ("artist", "artist_name", "creator", "author")),
            "album": _first_clean(row, ("album", "collection_name", "release", "album_name")),
            "description": _first_clean(
                row,
                (
                    "description",
                    "track_description",
                    "track_desc",
                    "track_summary",
                    "summary",
                    "notes",
                    "note",
                    "about",
                ),
            ),
            "album_description": _first_clean(
                row,
                (
                    "album_description",
                    "album_desc",
                    "album_summary",
                    "album_notes",
                    "collection_description",
                    "release_description",
                ),
            ),
            "description_source": _clean(row.get("description_source")),
            "album_description_source": _clean(row.get("album_description_source")),
            "description_evidence_url": _clean(row.get("description_evidence_url")),
            "culture": _clean(row.get("culture")),
            "source_dataset": _clean(row.get("source_dataset")),
            "label": _clean(row.get("label")),
            "label_en": _clean(row.get("label_en")),
            "tags": _clean(row.get("tags")),
            "tags_en": _clean(row.get("tags_en")),
            "description_en": _first_clean(row, ("description_en",)),
            "musicinfo_language": _clean(row.get("musicinfo_language")),
            "musicinfo_vocalinstrumental": _clean(row.get("musicinfo_vocalinstrumental")),
            "musicinfo_speed": _clean(row.get("musicinfo_speed")),
            "country": _clean(row.get("country")),
            "release_date": release_date,
            "release_year": release_year,
            "year": release_year,
            "era": _clean(row.get("era")),
            "duration_ms": _safe_float(row.get("duration_ms"), default=0.0),
            "audio_is_preview": _clean(row.get("audio_is_preview")),
            "preview_available": _clean(row.get("preview_available")),
            "cover_art_url": _clean(row.get("cover_art_url"))
            or _clean(row.get("artwork_url_large"))
            or _clean(row.get("image_url")),
            "cover_art_url_large": _clean(row.get("cover_art_url_large"))
            or _clean(row.get("cover_art_url"))
            or _clean(row.get("artwork_url_large")),
            "platform": _clean(row.get("platform")) or _clean(row.get("source_dataset")),
            "platform_track_url": _clean(row.get("platform_track_url"))
            or _clean(row.get("track_url"))
            or _clean(row.get("jamendo_url")),
            "platform_album_url": _clean(row.get("platform_album_url")) or _clean(row.get("collection_url")),
            "full_track_url": _clean(row.get("full_track_url")) or _clean(row.get("jamendo_url")),
            "preview_url": _clean(row.get("preview_url")) or _clean(row.get("audio_url")),
            "license_url": _clean(row.get("license_url")),
            "audio_api_url": f"/api/mainline/audio/{track_id}",
        }

    def _is_low_signal(self, row: dict[str, Any]) -> bool:
        text = " ".join(
            _clean(row.get(key)).lower() for key in ("title", "artist", "album", "label", "tags", "description")
        )
        return any(term in text for term in LOW_SIGNAL_TERMS)

    def _matches_query(self, row: dict[str, Any], query_terms: list[str]) -> bool:
        hay = _norm_key(
            " ".join(
                _clean(row.get(key))
                for key in (
                    "title",
                    "artist",
                    "album",
                    "label",
                    "tags",
                    "culture",
                    "source_dataset",
                    "country",
                )
            )
        )
        return all(term in hay for term in query_terms)


_CATALOGS: dict[tuple[str, str], LightweightMainlineCatalog] = {}


def get_lightweight_catalog(
    storage: Storage, *, metadata_rel: str = DEFAULT_METADATA_REL
) -> LightweightMainlineCatalog:
    key = (str(storage.root.resolve()), str(metadata_rel))
    catalog = _CATALOGS.get(key)
    if catalog is None:
        catalog = LightweightMainlineCatalog(storage=storage, metadata_rel=metadata_rel)
        _CATALOGS[key] = catalog
    return catalog
