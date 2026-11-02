"""Shared, hot-reloaded English translations keyed by the Chinese source text."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_load_lock = threading.RLock()

_next_check: dict[str, float] = {}

_cache: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}


def apply_description_translations(item: dict[str, Any], storage_root: Path) -> dict[str, Any]:
    path = Path(
        os.environ.get("ECHO_DESCRIPTION_TRANSLATIONS_PATH") or storage_root / "public/merged/descriptions_en.json"
    )
    key = str(path.absolute())
    with _load_lock:
        now = time.monotonic()
        if key not in _cache or now >= _next_check.get(key, 0):
            _next_check[key] = now + 0.5
            try:
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                cached = _cache.get(key)
                if cached is None or cached[0] != signature:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    translations = data["translations"]
                    if not isinstance(translations, dict) or not all(
                        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in translations.items()
                    ):
                        raise ValueError("invalid description dictionary")
                    _cache[key] = (signature, translations)
            except FileNotFoundError:
                _cache.pop(key, None)
            except (OSError, ValueError, KeyError):
                # Keep the last good version if a file is temporarily unreadable.
                pass
    translations = _cache.get(key, (None, {}))[1]
    for source_field in ("description", "album_description"):
        source = str(item.get(source_field) or "").strip()
        translated = translations.get(source)
        if translated:
            item[source_field + "_en"] = translated
    return item
