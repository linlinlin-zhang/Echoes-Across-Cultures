"""Apply source-backed artist geography separately from catalog market metadata."""

from __future__ import annotations
import json
import math
import threading
import time
from pathlib import Path
from typing import Any
import unicodedata

_load_lock = threading.RLock()

_next_check: dict[str, float] = {}

_cache: dict[str, tuple[tuple[int, int], dict]] = {}


def _load(path: Path) -> dict:
    with _load_lock:
        cache_key = str(path.absolute())
        now = time.monotonic()
        if cache_key in _cache and now < _next_check.get(cache_key, 0):
            return _cache[cache_key][1]
        _next_check[cache_key] = now + 0.5
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = _cache.get(cache_key)
            if cached is None or signature != cached[0]:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Invalid artist location dictionary")
                _cache[cache_key] = signature, data
        except FileNotFoundError:
            _cache.pop(cache_key, None)
        except (OSError, ValueError, KeyError):
            pass
        return _cache.get(cache_key, (None, {}))[1]


def apply_artist_location(item: dict[str, Any], root: Path) -> dict[str, Any]:
    cached_data = _load(root / "public/merged/artist_locations.json")
    artist_id = str(item.get("artist_id") or "").strip()
    key = (
        "itunes:" + artist_id
        if artist_id
        else "name:" + unicodedata.normalize("NFKC", str(item.get("artist") or "")).strip().casefold()
    )
    mapped_key = cached_data.get("tracks", {}).get(str(item.get("track_id") or item.get("id") or ""))
    artists = cached_data.get("artists", {})
    location = artists.get(mapped_key) or artists.get(key)
    if not isinstance(location, dict):
        fallback = _load(root / "public/merged/catalog_market_locations.json")
        location = fallback.get("tracks", {}).get(str(item.get("track_id") or item.get("id") or ""))
    if not isinstance(location, dict):
        return item
    try:
        lat, lng = float(location["lat"]), float(location["lng"])
        if not (math.isfinite(lat) and math.isfinite(lng) and -90 <= lat <= 90 and -180 <= lng <= 180):
            return item
        if not str(location.get("location_source_url") or "").startswith("https://"):
            return item
    except (KeyError, ValueError, TypeError):
        return item
    item["catalog_country"] = item.get("catalog_country", item.get("country", ""))
    for field in (
        "country",
        "city",
        "city_labels",
        "lat",
        "lng",
        "location_precision",
        "location_relation",
        "location_source_url",
        "location_place_url",
        "location_checked_at",
        "country_source",
    ):
        if field in location:
            item[field] = location[field]
    return item
