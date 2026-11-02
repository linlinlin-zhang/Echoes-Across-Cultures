"""Pass through real location metadata; never infer an artist's city from culture."""

from __future__ import annotations

import math
from typing import Any


def location_fields(row: dict[str, Any]) -> dict[str, Any]:
    def number(value: Any, limit: float) -> float | None:
        if value is None or isinstance(value, bool) or not str(value).strip():
            return None
        try:
            result = float(value)
        except (ValueError, TypeError):
            return None
        return result if math.isfinite(result) and abs(result) <= limit else None

    def first(*keys: str) -> Any:
        return next((row[key] for key in keys if row.get(key) is not None and str(row[key]).strip()), None)

    lat = number(first("lat", "latitude"), 90)
    lng = number(first("lng", "longitude", "lon"), 180)
    source = str(row.get("source_dataset") or row.get("platform") or "").lower()
    return {
        "city": str(row.get("city") or row.get("location_city") or row.get("artist_city") or "").strip(),
        "lat": lat if lat is not None and lng is not None else None,
        "lng": lng if lat is not None and lng is not None else None,
        "location_precision": "coordinates" if lat is not None and lng is not None else "",
        "country_source": str(row.get("country_source") or ("catalog_market" if source == "itunes" else "")),
    }
