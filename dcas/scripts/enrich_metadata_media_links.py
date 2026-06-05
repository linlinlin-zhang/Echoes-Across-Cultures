from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"

ENRICHED_FIELDS = [
    "cover_art_url",
    "cover_art_url_large",
    "platform",
    "platform_track_url",
    "platform_album_url",
    "platform_artist_url",
    "external_url",
    "full_track_url",
    "audio_is_preview",
    "preview_available",
    "track_url",
    "itunes_url",
    "apple_music_url",
    "spotify_url",
    "collection_url",
    "artist_url",
    "artist_id",
    "artwork_url_60",
    "artwork_url_large",
]


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _upscale_itunes_artwork(url: str, size: int = 600) -> str:
    if not url:
        return ""
    return re.sub(r"/\d+x\d+bb\.(jpg|png|webp)$", f"/{size}x{size}bb.\\1", url)


def _upscale_jamendo_image(url: str, size: int = 600) -> str:
    if not url:
        return ""
    if "width=" in url:
        return re.sub(r"([?&]width=)\d+", rf"\g<1>{size}", url)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}width={size}"


def _itunes_numeric_id(track_id: str) -> str:
    if track_id.startswith("itunes_"):
        return track_id.removeprefix("itunes_")
    return ""


def _lookup_itunes(
    rows: list[dict[str, str]],
    *,
    batch_size: int = 200,
    sleep_seconds: float = 0.4,
) -> dict[str, dict[str, Any]]:
    ids_by_country: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("source_dataset") != "itunes":
            continue
        tid = _itunes_numeric_id(str(row.get("track_id", "")))
        if not tid:
            continue
        if row.get("track_url") and row.get("collection_url") and row.get("artwork_url_large"):
            continue
        country = str(row.get("country") or "US").strip().upper() or "US"
        ids_by_country[country].append(tid)

    found: dict[str, dict[str, Any]] = {}
    for country, ids in sorted(ids_by_country.items()):
        unique_ids = sorted(set(ids))
        for start in range(0, len(unique_ids), batch_size):
            batch = unique_ids[start : start + batch_size]
            params = {
                "id": ",".join(batch),
                "country": country,
                "entity": "song",
            }
            for attempt in range(1, 4):
                try:
                    resp = requests.get(ITUNES_LOOKUP_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("results", []):
                        track_id = item.get("trackId")
                        if track_id:
                            found[f"itunes_{track_id}"] = item
                    break
                except requests.RequestException as exc:
                    if attempt >= 3:
                        print(f"[ITUNES LOOKUP ERROR] country={country} batch={start}: {exc}")
                    else:
                        time.sleep(2 * attempt)
            time.sleep(sleep_seconds)
    return found


def _first(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def enrich_metadata_media_links(
    metadata_csv: str | Path,
    out_csv: str | Path | None = None,
    *,
    itunes_lookup: bool = True,
    batch_size: int = 200,
) -> dict[str, Any]:
    in_path = Path(metadata_csv)
    out_path = Path(out_csv) if out_csv is not None else in_path
    rows, fields = _read_rows(in_path)

    itunes_info = _lookup_itunes(rows, batch_size=batch_size) if itunes_lookup else {}
    changed = 0
    itunes_linked = 0
    cover_filled = 0

    for row in rows:
        source = str(row.get("source_dataset", "")).strip().lower()
        if source == "itunes":
            item = itunes_info.get(str(row.get("track_id", "")), {})
            track_url = _first(row.get("track_url"), item.get("trackViewUrl"))
            collection_url = _first(row.get("collection_url"), item.get("collectionViewUrl"))
            artist_url = _first(row.get("artist_url"), item.get("artistViewUrl"))
            artwork_100 = _first(row.get("artwork_url"), item.get("artworkUrl100"))
            artwork_60 = _first(row.get("artwork_url_60"), item.get("artworkUrl60"))
            artwork_large = _first(row.get("artwork_url_large"), _upscale_itunes_artwork(artwork_100))

            updates = {
                "platform": "itunes",
                "track_url": track_url,
                "itunes_url": track_url,
                "apple_music_url": track_url,
                "collection_url": collection_url,
                "artist_url": artist_url,
                "artist_id": _first(row.get("artist_id"), str(item.get("artistId", "")) if item else ""),
                "artwork_url": artwork_100,
                "artwork_url_60": artwork_60,
                "artwork_url_large": artwork_large,
                "cover_art_url": _first(row.get("cover_art_url"), artwork_large, artwork_100, artwork_60),
                "cover_art_url_large": _first(row.get("cover_art_url_large"), artwork_large, artwork_100),
                "platform_track_url": _first(row.get("platform_track_url"), track_url),
                "platform_album_url": _first(row.get("platform_album_url"), collection_url),
                "platform_artist_url": _first(row.get("platform_artist_url"), artist_url),
                "external_url": _first(row.get("external_url"), track_url, collection_url),
                "full_track_url": _first(row.get("full_track_url"), track_url, collection_url),
                "audio_is_preview": "true",
                "preview_available": "true" if row.get("preview_url") else "",
            }
            if track_url:
                itunes_linked += 1
        elif source == "jamendo":
            image = _first(row.get("image_url"), row.get("cover_art_url"))
            large = _first(row.get("cover_art_url_large"), _upscale_jamendo_image(image))
            url = _first(row.get("jamendo_url"), row.get("external_url"))
            updates = {
                "platform": "jamendo",
                "cover_art_url": _first(row.get("cover_art_url"), image),
                "cover_art_url_large": large,
                "platform_track_url": _first(row.get("platform_track_url"), url),
                "platform_album_url": _first(row.get("platform_album_url"), ""),
                "platform_artist_url": _first(row.get("platform_artist_url"), ""),
                "external_url": _first(row.get("external_url"), url),
                "full_track_url": _first(row.get("full_track_url"), url, row.get("audio_url")),
                "audio_is_preview": "false",
                "preview_available": "false",
            }
        else:
            url = _first(row.get("spotify_url"), row.get("external_url"))
            updates = {
                "platform": source,
                "cover_art_url": _first(
                    row.get("cover_art_url"),
                    row.get("image_url"),
                    row.get("artwork_url"),
                ),
                "cover_art_url_large": _first(
                    row.get("cover_art_url_large"),
                    row.get("image_url"),
                    row.get("artwork_url"),
                ),
                "platform_track_url": _first(row.get("platform_track_url"), url),
                "external_url": _first(row.get("external_url"), url),
                "full_track_url": _first(row.get("full_track_url"), url),
            }

        before = dict(row)
        for key, value in updates.items():
            row[key] = str(value or "")
        if row.get("cover_art_url"):
            cover_filled += 1
        if row != before:
            changed += 1

    out_fields = list(fields)
    for field in ENRICHED_FIELDS:
        if field not in out_fields:
            out_fields.append(field)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in out_fields})
    tmp.replace(out_path)

    report = {
        "metadata": str(in_path.resolve()),
        "out_csv": str(out_path.resolve()),
        "rows": len(rows),
        "changed_rows": changed,
        "itunes_lookup_rows": len(itunes_info),
        "itunes_rows_with_track_url": itunes_linked,
        "rows_with_cover_art_url": cover_filled,
    }
    report_path = out_path.with_suffix(out_path.suffix + ".media_links_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Add cover-art and playable platform links to merged music metadata.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--no_itunes_lookup", action="store_true")
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()

    report = enrich_metadata_media_links(
        metadata_csv=args.metadata,
        out_csv=args.out or None,
        itunes_lookup=not args.no_itunes_lookup,
        batch_size=int(args.batch_size),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
