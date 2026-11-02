"""Rank direct song/artist matches ahead of incidental catalog tag matches."""

from __future__ import annotations
from typing import Any


def query_rank(row: dict[str, Any], query: str) -> int:
    def normalize(value: Any) -> str:
        return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())

    fields = [normalize(row.get(key)) for key in ("title", "artist")]
    if query in fields:
        return 0
    if any(field.startswith(query) for field in fields):
        return 1
    if any(query in field for field in fields):
        return 2
    if all(term in " ".join(fields) for term in query.split()):
        return 3
    return 4
