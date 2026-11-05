"""Shared genre identity, localized names and explicit non-music metadata.

The raw source CSV and model features remain unchanged. This layer normalizes API
metadata and search indexes using the same reviewed dictionary as the browser.
"""

from __future__ import annotations
from functools import lru_cache
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[1] / "web/genre-data.json"


def normalize_key(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).lower().replace("’", "'").split())


@lru_cache(maxsize=1)
def _data() -> dict:
    if not DATA_PATH.exists():
        return {"aliases": {}, "genres": {}}
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=4096)
def genre_for_label(label: str) -> tuple[str, dict] | None:
    data = _data()
    gid = data["aliases"].get(normalize_key(label))
    return (gid, data["genres"][gid]) if gid else None


def raw_label(item: dict[str, Any]) -> str:
    return str(
        item.get("label_original") or item.get("genre_original") or item.get("label") or item.get("genre") or ""
    ).strip()


def is_non_music(item: dict[str, Any]) -> bool:
    genre = genre_for_label(raw_label(item))
    return bool(genre and genre[1].get("exclude_from_music"))


def genre_search_text(item: dict[str, Any]) -> str:
    genre = genre_for_label(raw_label(item))
    return " ".join(genre[1]["labels"].values()) if genre else ""


def normalize_tags(value: str) -> str:
    exact = genre_for_label(value)
    if exact:
        return exact[1]["labels"]["en"]
    parts = re.split(r"[;,|]", value)
    output = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        found = genre_for_label(part)
        token = found[1]["labels"]["en"] if found else part
        # A field named *_en must not silently carry unrecognized East Asian text.
        if not found and re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", token):
            continue
        if token not in output:
            output.append(token)
    return "; ".join(output)


def apply_genre_metadata(item: dict[str, Any]) -> dict[str, Any]:
    original = raw_label(item)
    found = genre_for_label(original)
    if original:
        item.setdefault("label_original", original)
    item["genre_normalized"] = bool(found)
    if found:
        gid, genre = found
        english = genre["labels"]["en"]
        item.setdefault("label_en_original", str(item.get("label_en") or ""))
        item.update(
            label=english,
            label_en=english,
            genre_id=gid,
            genre_labels=dict(genre["labels"]),
            genre_kind=genre["kind"],
            content_kind="spoken_drama" if genre.get("exclude_from_music") else "music",
        )
        if "genre" in item:
            item["genre"] = english
            item["genre_en"] = english
    elif re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", str(item.get("label_en") or "")):
        item.setdefault("label_en_original", str(item["label_en"]))
        item["label_en"] = ""
    tags = str(item.get("tags_original", item.get("tags") or ""))
    if tags:
        item.setdefault("tags_original", tags)
        item.setdefault("tags_en_original", str(item.get("tags_en") or ""))
        item["tags_en"] = normalize_tags(tags)
    if "duration_ms" in item:
        try:
            value = float(item["duration_ms"])
            known = 0 < value < float("inf")
        except (ValueError, TypeError):
            known = False
        item["duration_known"] = known
        if not known:
            item["duration_ms"] = None
    return item
