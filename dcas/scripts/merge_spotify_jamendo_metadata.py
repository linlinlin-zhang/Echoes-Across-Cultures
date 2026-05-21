"""
Merge and align Spotify + Jamendo metadata for unified DCAS pipeline.

Produces a single metadata_merged.csv with consistent columns required by
build_tracks_from_audio.py, plus enriched fields from both sources.

Usage:
    python -m dcas.scripts.merge_spotify_jamendo_metadata \
        --spotify ./storage/public/spotify_crawl/metadata.csv \
        --jamendo ./storage/public/jamendo_crawl/metadata.csv \
        --out ./storage/public/merged/metadata_merged.csv \
        --dedup_key track_id

    # If you have multiple shards from distributed crawls:
    python -m dcas.scripts.merge_spotify_jamendo_metadata \
        --spotify "./storage/public/spotify_*/metadata.csv" \
        --jamendo "./storage/public/jamendo_*/metadata.csv" \
        --out ./storage/public/merged/metadata_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Unified schema
# ---------------------------------------------------------------------------

# Columns required by build_tracks_from_audio.py
REQUIRED_COLS = ["track_id", "culture", "audio_path"]

# Unified output columns (superset of both sources)
OUTPUT_COLUMNS = [
    "track_id",
    "culture",
    "audio_path",
    "label",
    "title",
    "artist",
    "album",
    "duration_ms",
    "popularity",
    "explicit",
    # Spotify audio features (may be empty for Jamendo)
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    # Source provenance
    "source",
    "source_market",
    # Jamendo-specific (empty for Spotify)
    "jamendo_id",
    "jamendo_url",
    "audio_url",
    "image_url",
    "license_url",
    "tags",
    # Spotify-specific (empty for Jamendo)
    "spotify_preview_url",
]


# ---------------------------------------------------------------------------
# Culture normalization
# ---------------------------------------------------------------------------

CULTURE_ALIASES: dict[str, str] = {
    # Standard mappings
    "western": "west",
    "occidental": "west",
    "euro-american": "west",
    "european": "west",
    "american": "west",
    "usa": "west",
    "uk": "west",
    "british": "west",
    "korean": "korea",
    "kr": "korea",
    "south_korea": "korea",
    "japanese": "japan",
    "jp": "japan",
    "chinese": "china",
    "cn": "china",
    "mandarin": "china",
    "cantonese": "china",
    "indian": "india",
    "in": "india",
    "hindustani": "india",
    "carnatic": "india",
    "brazilian": "brazil",
    "br": "brazil",
    "latin_american": "latin",
    "latino": "latin",
    "hispanic": "latin",
    "es": "latin",
    "mx": "latin",
    "african": "africa",
    "afro": "africa",
    "arabic": "middle_east",
    "arab": "middle_east",
    "persian": "middle_east",
    "iranian": "middle_east",
    "turkish": "middle_east",
    "turkey": "middle_east",
    "middle_eastern": "middle_east",
    "southeast_asian": "southeast_asia",
    "sea": "southeast_asia",
    "thai": "southeast_asia",
    "vietnamese": "southeast_asia",
    "philippine": "southeast_asia",
    "malay": "southeast_asia",
    "indonesian": "southeast_asia",
    "celtic": "celtic",
    "irish": "celtic",
    "scottish": "celtic",
}


def normalize_culture(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]", "_", str(raw).strip().lower())
    return CULTURE_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Row normalization
# ---------------------------------------------------------------------------

def normalize_spotify_row(row: dict[str, str]) -> dict[str, str]:
    """Convert a Spotify metadata row into the unified schema."""
    out: dict[str, str] = {col: "" for col in OUTPUT_COLUMNS}

    out["track_id"] = str(row.get("track_id", "")).strip()
    out["culture"] = normalize_culture(row.get("culture", ""))
    out["audio_path"] = str(row.get("audio_path", "")).strip()
    out["label"] = str(row.get("label", "")).strip()
    out["title"] = str(row.get("title", "")).strip()
    out["artist"] = str(row.get("artist", "")).strip()
    out["album"] = str(row.get("album", "")).strip()
    out["duration_ms"] = str(row.get("duration_ms", "")).strip()
    out["popularity"] = str(row.get("popularity", "0")).strip()
    out["explicit"] = str(row.get("explicit", "False")).strip()

    # Audio features
    for af in (
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "time_signature",
    ):
        out[af] = str(row.get(af, "")).strip()

    out["source"] = "spotify"
    out["source_market"] = str(row.get("market", "")).strip()
    out["spotify_preview_url"] = str(row.get("preview_url", "")).strip()
    return out


def normalize_jamendo_row(row: dict[str, str]) -> dict[str, str]:
    """Convert a Jamendo metadata row into the unified schema."""
    out: dict[str, str] = {col: "" for col in OUTPUT_COLUMNS}

    out["track_id"] = str(row.get("track_id", "")).strip()
    out["culture"] = normalize_culture(row.get("culture", ""))
    out["audio_path"] = str(row.get("audio_path", "")).strip()
    out["label"] = str(row.get("label", "")).strip()
    out["title"] = str(row.get("title", "")).strip()
    out["artist"] = str(row.get("artist", "")).strip()
    out["album"] = str(row.get("album", "")).strip()
    out["duration_ms"] = str(row.get("duration_ms", "")).strip()
    out["popularity"] = ""
    out["explicit"] = ""

    # Jamendo specifics
    out["jamendo_id"] = str(row.get("jamendo_id", "")).strip()
    out["jamendo_url"] = str(row.get("jamendo_url", "")).strip()
    out["audio_url"] = str(row.get("audio_url", "")).strip()
    out["image_url"] = str(row.get("image_url", "")).strip()
    out["license_url"] = str(row.get("license_url", "")).strip()
    out["tags"] = str(row.get("tags", "")).strip()

    out["source"] = "jamendo"
    out["source_market"] = ""
    return out


# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------

def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns to concrete paths."""
    paths: list[Path] = []
    for p in patterns:
        pobj = Path(p)
        if "*" in str(pobj) or "?" in str(pobj):
            # Use parent dir glob
            parent = pobj.parent
            if not parent.exists():
                continue
            matched = list(parent.glob(pobj.name))
            paths.extend(matched)
        else:
            if pobj.exists():
                paths.append(pobj)
    return paths


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: str(v) if v is not None else "" for k, v in row.items()})
    return rows


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_metadata(
    spotify_paths: list[Path],
    jamendo_paths: list[Path],
    out_path: Path,
    dedup_key: str = "track_id",
    require_audio_exists: bool = False,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys: set[str] = set()
    merged_rows: list[dict[str, str]] = []
    stats = {
        "spotify_files": len(spotify_paths),
        "jamendo_files": len(jamendo_paths),
        "spotify_raw_rows": 0,
        "jamendo_raw_rows": 0,
        "spotify_deduped": 0,
        "jamendo_deduped": 0,
        "total_output": 0,
        "by_source": {"spotify": 0, "jamendo": 0},
        "by_culture": {},
        "missing_required": 0,
        "missing_audio_file": 0,
    }

    # Process Spotify
    for sp in spotify_paths:
        raw_rows = read_csv_rows(sp)
        stats["spotify_raw_rows"] += len(raw_rows)
        for row in raw_rows:
            norm = normalize_spotify_row(row)
            key = norm.get(dedup_key, "").strip()
            if not key:
                stats["missing_required"] += 1
                continue
            if require_audio_exists:
                audio_rel = norm.get("audio_path", "")
                if audio_rel:
                    audio_abs = sp.parent / audio_rel
                    if not audio_abs.exists():
                        stats["missing_audio_file"] += 1
                        continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_rows.append(norm)
            stats["spotify_deduped"] += 1
            stats["by_source"]["spotify"] += 1
            cul = norm.get("culture", "unknown")
            stats["by_culture"][cul] = stats["by_culture"].get(cul, 0) + 1

    # Process Jamendo
    for jp in jamendo_paths:
        raw_rows = read_csv_rows(jp)
        stats["jamendo_raw_rows"] += len(raw_rows)
        for row in raw_rows:
            norm = normalize_jamendo_row(row)
            key = norm.get(dedup_key, "").strip()
            if not key:
                stats["missing_required"] += 1
                continue
            if require_audio_exists:
                audio_rel = norm.get("audio_path", "")
                if audio_rel:
                    audio_abs = jp.parent / audio_rel
                    if not audio_abs.exists():
                        stats["missing_audio_file"] += 1
                        continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_rows.append(norm)
            stats["jamendo_deduped"] += 1
            stats["by_source"]["jamendo"] += 1
            cul = norm.get("culture", "unknown")
            stats["by_culture"][cul] = stats["by_culture"].get(cul, 0) + 1

    stats["total_output"] = len(merged_rows)

    # Write merged CSV
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    # Write report
    report_path = out_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[MERGE DONE] Output: {out_path}")
    print(f"  Spotify: {stats['spotify_deduped']} / {stats['spotify_raw_rows']} (after dedup)")
    print(f"  Jamendo: {stats['jamendo_deduped']} / {stats['jamendo_raw_rows']} (after dedup)")
    print(f"  Total unique: {stats['total_output']}")
    print(f"  By culture: {stats['by_culture']}")
    print(f"  Report: {report_path}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Merge and align Spotify + Jamendo metadata.")
    ap.add_argument("--spotify", nargs="+", default=[], help="Path(s) to Spotify metadata.csv (supports globs)")
    ap.add_argument("--jamendo", nargs="+", default=[], help="Path(s) to Jamendo metadata.csv (supports globs)")
    ap.add_argument("--out", required=True, help="Output merged metadata_merged.csv path")
    ap.add_argument("--dedup_key", default="track_id", help="Column to use for deduplication")
    ap.add_argument("--require_audio_exists", action="store_true", help="Skip rows whose audio file does not exist on disk")
    args = ap.parse_args()

    if not args.spotify and not args.jamendo:
        print("[ERROR] At least one of --spotify or --jamendo must be provided.")
        sys.exit(1)

    spotify_paths = expand_paths(args.spotify)
    jamendo_paths = expand_paths(args.jamendo)

    print(f"[INFO] Spotify files: {len(spotify_paths)}")
    for sp in spotify_paths:
        print(f"   - {sp}")
    print(f"[INFO] Jamendo files: {len(jamendo_paths)}")
    for jp in jamendo_paths:
        print(f"   - {jp}")

    merge_metadata(
        spotify_paths=spotify_paths,
        jamendo_paths=jamendo_paths,
        out_path=Path(args.out),
        dedup_key=args.dedup_key,
        require_audio_exists=args.require_audio_exists,
    )


if __name__ == "__main__":
    main()
