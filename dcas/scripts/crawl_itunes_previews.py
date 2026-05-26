"""
Apple iTunes Search API Preview Crawler for DCAS

Collects 30-second AAC preview files from Apple iTunes Search API.
No API key required. Rate limit is approximately 20 requests/minute.

Output metadata.csv is compatible with build_tracks_from_audio.py.

Usage:
    # Start fresh
    python -m dcas.scripts.crawl_itunes_previews \
        --out_dir ./storage/public/itunes_crawl \
        --target_total 5000 \
        --workers 4

    # Resume interrupted crawl
    python -m dcas.scripts.crawl_itunes_previews \
        --out_dir ./storage/public/itunes_crawl \
        --resume

    # Target specific countries
    python -m dcas.scripts.crawl_itunes_previews \
        --out_dir ./storage/public/itunes_crawl \
        --countries US,JP,KR,GB,BR,MX,IN,FR,DE \
        --target_total 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests


ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"

# Apple rate limit: ~20 req/min for Search API
REQUEST_INTERVAL = 3.5  # seconds between requests (conservative)


def _upscale_itunes_artwork(url: str, size: int = 600) -> str:
    if not url:
        return ""
    return re.sub(r"/\d+x\d+bb\.(jpg|png|webp)$", f"/{size}x{size}bb.\\1", url)

# ---------------------------------------------------------------------------
# Country to culture mapping
# ---------------------------------------------------------------------------

COUNTRY_TO_CULTURE: dict[str, str] = {
    "US": "west", "GB": "west", "CA": "west", "AU": "west", "NZ": "west",
    "DE": "west", "FR": "west", "ES": "west", "NL": "west", "SE": "west",
    "IT": "west", "PL": "west", "IE": "west", "NO": "west", "FI": "west",
    "DK": "west", "AT": "west", "CH": "west", "BE": "west", "PT": "west",
    "JP": "japan",
    "KR": "korea",
    "IN": "india",
    "TW": "china", "HK": "china", "SG": "china",
    "CN": "china",
    "BR": "brazil",
    "MX": "latin", "CO": "latin", "CL": "latin", "AR": "latin", "PE": "latin",
    "VE": "latin", "EC": "latin", "UY": "latin",
    "ZA": "africa", "NG": "africa", "EG": "africa", "GH": "africa", "KE": "africa",
    "TZ": "africa", "UG": "africa",
    "TR": "middle_east", "IL": "middle_east", "SA": "middle_east", "AE": "middle_east",
    "QA": "middle_east", "KW": "middle_east", "BH": "middle_east", "OM": "middle_east",
    "JO": "middle_east", "LB": "middle_east", "IQ": "middle_east", "IR": "middle_east",
    "ID": "southeast_asia", "TH": "southeast_asia", "PH": "southeast_asia",
    "MY": "southeast_asia", "VN": "southeast_asia", "KH": "southeast_asia",
    "LA": "southeast_asia", "MM": "southeast_asia", "BN": "southeast_asia",
}

# Search terms to rotate through per country
SEARCH_TERMS = [
    "pop", "rock", "hip hop", "electronic", "dance", "indie",
    "soul", "rnb", "jazz", "classical", "folk", "country",
    "alternative", "metal", "punk", "reggae", "blues",
    "top", "hit", "chart", "single", "album",
]

ERA_SEARCH_TERMS = [
    "1950s music", "1960s music", "1970s music", "1980s music",
    "1990s music", "2000s music", "2010s music", "2020s music",
    "oldies", "classic hits",
]

CULTURE_SEARCH_TERMS: dict[str, list[str]] = {
    "west": ["oldies", "classic rock", "singer songwriter", "eurovision"],
    "japan": ["j-pop", "j-rock", "city pop", "enka", "anime"],
    "korea": ["k-pop", "korean", "trot", "k-indie"],
    "india": ["bollywood", "indian pop", "hindustani", "carnatic"],
    "china": [
        "mandopop", "cantopop", "chinese pop", "taiwan pop",
        "cantonese songs", "hong kong pop", "hakka songs", "hokkien songs",
        "taiwanese hokkien", "minnan songs", "teochew songs",
        "shanghainese songs", "sichuan dialect songs", "wu chinese songs",
        "yue chinese songs",
    ],
    "brazil": ["samba", "bossa nova", "mpb", "sertanejo"],
    "latin": ["reggaeton", "salsa", "bachata", "cumbia", "tango"],
    "africa": ["afrobeats", "amapiano", "highlife", "soukous"],
    "middle_east": ["arabic pop", "turkish pop", "persian pop", "rai"],
    "southeast_asia": ["thai pop", "dangdut", "v-pop", "opm"],
    "celtic": ["celtic", "irish folk", "scottish folk", "gaelic", "breton", "fiddle"],
    "nordic": ["nordic", "scandinavian", "swedish pop", "norwegian", "finnish", "danish", "icelandic"],
    "eastern_europe": ["polish folk", "ukrainian", "czech", "hungarian", "romanian", "slavic"],
    "balkans": ["balkan", "greek", "serbian", "croatian", "bulgarian", "sevdah", "turbofolk"],
    "caribbean": ["caribbean", "reggae", "dancehall", "soca", "calypso", "zouk", "kompa"],
    "andean": ["andean", "huayno", "quechua music", "charango", "peruvian folk", "bolivian folk"],
    "central_asia": ["central asian", "kazakh", "uzbek", "kyrgyz", "tajik", "turkmen"],
}

DEFAULT_COUNTRIES = list(COUNTRY_TO_CULTURE.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackRecord:
    track_id: str
    title: str
    artist: str
    album: str
    genre: str
    culture: str
    country: str
    preview_url: str
    duration_ms: int
    release_date: str
    explicit: str
    artwork_url: str
    artwork_url_60: str
    artwork_url_large: str
    collection_id: str
    artist_id: str
    track_url: str
    collection_url: str
    artist_url: str

    def to_metadata_row(self, audio_rel_path: str) -> dict[str, str]:
        return {
            "track_id": self.track_id,
            "culture": self.culture,
            "audio_path": audio_rel_path,
            "source_dataset": "itunes",
            "label": self.genre,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "country": self.country,
            "duration_ms": str(self.duration_ms),
            "explicit": self.explicit,
            "release_date": self.release_date,
            "preview_url": self.preview_url,
            "artwork_url": self.artwork_url,
            "artwork_url_60": self.artwork_url_60,
            "artwork_url_large": self.artwork_url_large,
            "collection_id": self.collection_id,
            "artist_id": self.artist_id,
            "track_url": self.track_url,
            "itunes_url": self.track_url,
            "apple_music_url": self.track_url,
            "collection_url": self.collection_url,
            "artist_url": self.artist_url,
        }


@dataclass
class CrawlState:
    version: int = 1
    completed_queries: list[str] = field(default_factory=list)
    downloaded_track_ids: list[str] = field(default_factory=list)
    failed_track_ids: list[str] = field(default_factory=list)
    total_collected: int = 0
    total_downloaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrawlState:
        return cls(
            version=d.get("version", 1),
            completed_queries=list(d.get("completed_queries", [])),
            downloaded_track_ids=list(d.get("downloaded_track_ids", [])),
            failed_track_ids=list(d.get("failed_track_ids", [])),
            total_collected=int(d.get("total_collected", 0)),
            total_downloaded=int(d.get("total_downloaded", 0)),
        )


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.state_path = out_dir / "state.json"
        self.metadata_path = out_dir / "metadata.csv"

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> CrawlState:
        with open(self.state_path, "r", encoding="utf-8") as f:
            return CrawlState.from_dict(json.load(f))

    def save(self, state: CrawlState) -> None:
        tmp = self.state_path.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        tmp.replace(self.state_path)

    def append_metadata_rows(self, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
        write_header = not self.metadata_path.exists()
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
                existing_rows = list(reader)
            if existing_fields and existing_fields != fieldnames:
                merged_fields = list(existing_fields)
                for name in fieldnames:
                    if name not in merged_fields:
                        merged_fields.append(name)
                tmp = self.metadata_path.with_suffix(".tmp.csv")
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=merged_fields)
                    writer.writeheader()
                    for row in existing_rows:
                        writer.writerow({c: str(row.get(c, "")) for c in merged_fields})
                tmp.replace(self.metadata_path)
                fieldnames = merged_fields
                write_header = False
        with open(self.metadata_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def read_existing_metadata_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self.metadata_path.exists():
            return ids
        with open(self.metadata_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = str(row.get("track_id", "")).strip()
                if tid:
                    ids.add(tid)
        return ids


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class RateLimitedSession:
    def __init__(self, interval: float = REQUEST_INTERVAL, max_retries: int = 4, backoff: float = 5.0):
        self.interval = interval
        self.max_retries = max(1, int(max_retries))
        self.backoff = max(0.1, float(backoff))
        self._last_request_time: float = 0.0

    def _wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request_time = time.time()

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._wait()
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 403:
                    print("[RATE LIMIT] 403 from iTunes; backing off 60s")
                    time.sleep(60)
                    self._last_request_time = time.time()
                    return {}
                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_secs = self.backoff * attempt
                    print(f"[REQUEST RETRY] status={resp.status_code}; sleeping {sleep_secs:.1f}s ({attempt}/{self.max_retries})")
                    time.sleep(sleep_secs)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_secs = self.backoff * attempt
                print(f"[REQUEST RETRY] {type(exc).__name__}: {exc}; sleeping {sleep_secs:.1f}s ({attempt}/{self.max_retries})")
                time.sleep(sleep_secs)
        if last_error is not None:
            raise last_error
        return {}


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class iTunesCrawler:
    def __init__(
        self,
        session: RateLimitedSession,
        checkpoint: CheckpointManager,
        out_dir: Path,
        countries: list[str],
        target_total: int,
        workers: int,
        checkpoint_interval: int,
        max_per_query: int,
        culture_override: str = "",
        extra_terms: list[str] | None = None,
    ):
        self.session = session
        self.checkpoint = checkpoint
        self.out_dir = out_dir
        self.countries = countries
        self.target_total = target_total
        self.workers = workers
        self.checkpoint_interval = checkpoint_interval
        self.max_per_query = max(1, int(max_per_query))
        self.culture_override = culture_override.strip()
        self.extra_terms = [t.strip() for t in (extra_terms or []) if t.strip()]

        self.audio_dir = out_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._fieldnames = [
            "track_id", "culture", "audio_path", "source_dataset", "label",
            "title", "artist", "album", "country",
            "duration_ms", "explicit", "release_date",
            "preview_url", "artwork_url", "artwork_url_60", "artwork_url_large",
            "collection_id", "artist_id", "track_url", "itunes_url", "apple_music_url",
            "collection_url", "artist_url",
        ]

    def search_tracks(
        self,
        country: str,
        term: str,
        offset: int = 0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "term": term,
            "country": country,
            "media": "music",
            "entity": "song",
            "limit": limit,
            "offset": offset,
        }
        data = self.session.get(ITUNES_SEARCH_URL, params=params)
        return list(data.get("results", []))

    @staticmethod
    def _parse_track(
        item: dict[str, Any],
        culture: str,
        country: str,
        label: str,
    ) -> TrackRecord | None:
        tid = str(item.get("trackId", ""))
        if not tid:
            return None
        preview = item.get("previewUrl", "")
        if not preview:
            return None
        artwork_url = str(item.get("artworkUrl100", ""))
        return TrackRecord(
            track_id=f"itunes_{tid}",
            title=str(item.get("trackName", "")),
            artist=str(item.get("artistName", "Unknown")),
            album=str(item.get("collectionName", "")),
            genre=str(item.get("primaryGenreName", label)),
            culture=culture,
            country=country,
            preview_url=preview,
            duration_ms=int(item.get("trackTimeMillis", 0)),
            release_date=str(item.get("releaseDate", "")),
            explicit=str(item.get("trackExplicitness", "notExplicit")),
            artwork_url=artwork_url,
            artwork_url_60=str(item.get("artworkUrl60", "")),
            artwork_url_large=_upscale_itunes_artwork(artwork_url),
            collection_id=str(item.get("collectionId", "")),
            artist_id=str(item.get("artistId", "")),
            track_url=str(item.get("trackViewUrl", "")),
            collection_url=str(item.get("collectionViewUrl", "")),
            artist_url=str(item.get("artistViewUrl", "")),
        )

    def download_preview(self, record: TrackRecord) -> Path | None:
        safe_tid = re.sub(r"[^a-zA-Z0-9._-]", "_", record.track_id)
        safe_culture = re.sub(r"[^a-zA-Z0-9._-]", "_", record.culture)[:32]
        safe_country = re.sub(r"[^a-zA-Z0-9._-]", "_", record.country)[:32]
        subdir = self.audio_dir / safe_culture / safe_country
        subdir.mkdir(parents=True, exist_ok=True)
        dest = subdir / f"{safe_tid}.m4a"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest
        try:
            resp = requests.get(record.preview_url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if dest.stat().st_size < 1024:
                dest.unlink()
                return None
            return dest
        except Exception as e:
            print(f"[DOWNLOAD ERROR] {record.track_id}: {e}")
            if dest.exists():
                dest.unlink(missing_ok=True)
            return None

    def _build_query_pool(self) -> list[tuple[str, str, str]]:
        pool: list[tuple[str, str, str]] = []
        for country in self.countries:
            culture = self.culture_override or COUNTRY_TO_CULTURE.get(country, "west")
            # Empty term search not supported well; use genre terms
            terms = SEARCH_TERMS + ERA_SEARCH_TERMS + CULTURE_SEARCH_TERMS.get(culture, []) + self.extra_terms
            seen_terms: set[str] = set()
            for term in terms:
                if term in seen_terms:
                    continue
                seen_terms.add(term)
                pool.append((country, term, culture))
            # Add some country-specific terms
            if culture == "korea":
                pool.append((country, "k-pop", culture))
                pool.append((country, "korean", culture))
            elif culture == "japan":
                pool.append((country, "j-pop", culture))
                pool.append((country, "anime", culture))
            elif culture == "india":
                pool.append((country, "bollywood", culture))
                pool.append((country, "indian pop", culture))
            elif culture == "brazil":
                pool.append((country, "samba", culture))
                pool.append((country, "bossa nova", culture))
            elif culture == "latin":
                pool.append((country, "reggaeton", culture))
                pool.append((country, "salsa", culture))
            elif culture == "china":
                pool.append((country, "mandopop", culture))
                pool.append((country, "cantopop", culture))
            elif culture == "africa":
                pool.append((country, "afrobeats", culture))
                pool.append((country, "amapiano", culture))
            elif culture == "middle_east":
                pool.append((country, "arabic pop", culture))
                pool.append((country, "turkish pop", culture))
            elif culture == "southeast_asia":
                pool.append((country, "thai pop", culture))
                pool.append((country, "dangdut", culture))
        random.shuffle(pool)
        return pool

    def run(self, resume: bool = False) -> dict[str, Any]:
        state = CrawlState()
        if resume and self.checkpoint.exists():
            state = self.checkpoint.load()
            print(f"[RESUME] collected={state.total_collected}, downloaded={state.total_downloaded}")
        else:
            if self.metadata_path.exists() and not resume:
                print("[WARN] metadata.csv exists but --resume not set. Starting fresh.")

        completed_set = set(state.completed_queries)
        downloaded_set = set(state.downloaded_track_ids)
        failed_set = set(state.failed_track_ids)
        existing_meta_ids = self.checkpoint.read_existing_metadata_ids()
        downloaded_set |= existing_meta_ids
        if existing_meta_ids:
            state.downloaded_track_ids = sorted(downloaded_set)
            if len(downloaded_set) > state.total_downloaded:
                state.total_downloaded = len(downloaded_set)
            synced_collected = len(downloaded_set) + len(failed_set)
            if synced_collected > state.total_collected:
                state.total_collected = synced_collected
        if resume and state.completed_queries and not downloaded_set and not failed_set:
            print("[RECOVERY] Existing checkpoint has completed queries but no downloaded/failed tracks; retrying queries.")
            state.completed_queries.clear()
            state.total_collected = 0
            completed_set.clear()
        seen_set = set(downloaded_set) | set(failed_set)
        print(f"[INIT] {len(downloaded_set)} tracks already in metadata.csv")

        query_pool = self._build_query_pool()
        total_queries = len(query_pool)
        print(f"[INIT] Query pool: {total_queries} queries across {len(self.countries)} countries")

        last_checkpoint_time = time.time()
        pending_metadata_rows: list[dict[str, str]] = []

        def do_checkpoint(force: bool = False) -> None:
            nonlocal last_checkpoint_time, pending_metadata_rows
            now = time.time()
            if not force and (now - last_checkpoint_time) < self.checkpoint_interval:
                return
            if pending_metadata_rows:
                self.checkpoint.append_metadata_rows(pending_metadata_rows, self._fieldnames)
                pending_metadata_rows.clear()
            self.checkpoint.save(state)
            last_checkpoint_time = now
            print(f"[CHECKPOINT] collected={state.total_collected}, downloaded={state.total_downloaded}")

        def process_download_batch(records: list[TrackRecord]) -> None:
            nonlocal pending_metadata_rows
            to_download = [r for r in records if r.track_id not in downloaded_set and r.track_id not in failed_set]
            if not to_download:
                return
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {ex.submit(self.download_preview, r): r for r in to_download}
                for fut in as_completed(futures):
                    rec = futures[fut]
                    try:
                        path = fut.result()
                        if path:
                            rel = path.relative_to(self.out_dir).as_posix()
                            pending_metadata_rows.append(rec.to_metadata_row(rel))
                            downloaded_set.add(rec.track_id)
                            state.downloaded_track_ids.append(rec.track_id)
                            state.total_downloaded += 1
                        else:
                            failed_set.add(rec.track_id)
                            state.failed_track_ids.append(rec.track_id)
                    except Exception as e:
                        print(f"[BATCH ERROR] {rec.track_id}: {e}")
                        failed_set.add(rec.track_id)
                        state.failed_track_ids.append(rec.track_id)

        try:
            for qidx, (country, term, culture) in enumerate(query_pool, start=1):
                query_key = f"{culture}::{country}::{term}" if self.culture_override else f"{country}::{term}"
                if query_key in completed_set:
                    continue

                if state.total_collected >= self.target_total:
                    print(f"[TARGET REACHED] {state.total_collected} >= {self.target_total}")
                    break

                print(f"[{qidx}/{total_queries}] country={country} term={term}")

                all_records: list[TrackRecord] = []
                remaining = max(0, self.target_total - state.total_collected)
                query_limit = min(remaining, self.max_per_query)
                query_failed = False
                for offset in range(0, 2000, 200):
                    try:
                        items = self.search_tracks(country, term, offset=offset)
                    except requests.RequestException as exc:
                        query_failed = True
                        print(
                            f"[QUERY ERROR] country={country} term={term!r} "
                            f"offset={offset}: {type(exc).__name__}: {exc}"
                        )
                        break
                    if not items:
                        break
                    for item in items:
                        rec = self._parse_track(item, culture, country, term)
                        if rec and rec.track_id not in seen_set:
                            all_records.append(rec)
                            seen_set.add(rec.track_id)
                            if len(all_records) >= query_limit:
                                break
                    if len(all_records) >= query_limit:
                        break
                    if len(items) < 200:
                        break
                    if state.total_collected + len(all_records) >= self.target_total:
                        break

                if query_failed:
                    do_checkpoint(force=True)
                    continue

                if not all_records:
                    completed_set.add(query_key)
                    state.completed_queries.append(query_key)
                    do_checkpoint()
                    continue

                state.total_collected += len(all_records)
                print(f"  -> new records={len(all_records)}")

                process_download_batch(all_records)

                completed_set.add(query_key)
                state.completed_queries.append(query_key)
                do_checkpoint(force=True)

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Saving checkpoint...")
        finally:
            do_checkpoint(force=True)

        report = {
            "out_dir": str(self.out_dir.resolve()),
            "target_total": self.target_total,
            "total_collected": state.total_collected,
            "total_downloaded": state.total_downloaded,
            "completed_queries": len(state.completed_queries),
            "failed_tracks": len(state.failed_track_ids),
            "metadata_csv": str(self.checkpoint.metadata_path.resolve()),
        }
        report_path = self.out_dir / "import_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[DONE] {report_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    @property
    def metadata_path(self) -> Path:
        return self.checkpoint.metadata_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk collect Apple iTunes track previews for DCAS.")
    ap.add_argument("--out_dir", required=True, help="Output directory for audio + metadata")
    ap.add_argument("--countries", default="", help="Comma-separated ISO country codes (default: all)")
    ap.add_argument("--target_total", type=int, default=5_000, help="Target unique tracks")
    ap.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    ap.add_argument("--checkpoint_interval", type=int, default=300, help="Seconds between checkpoints")
    ap.add_argument("--max_per_query", type=int, default=50, help="Max new tracks downloaded from a single country/term query")
    ap.add_argument("--culture_override", default="", help="Force all downloaded rows to this culture label")
    ap.add_argument("--extra_terms", default="", help="Comma-separated extra search terms to append to the query pool")
    ap.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    if not countries:
        countries = list(DEFAULT_COUNTRIES)
    else:
        invalid = [c for c in countries if c not in COUNTRY_TO_CULTURE]
        if invalid and not args.culture_override.strip():
            print(f"[WARN] Unknown countries (will use 'west'): {invalid}")

    extra_terms = [t.strip() for t in args.extra_terms.split(",") if t.strip()]

    session = RateLimitedSession()
    checkpoint = CheckpointManager(out_dir)

    # Quick API test
    test_data = session.get(ITUNES_SEARCH_URL, params={"term": "test", "media": "music", "entity": "song", "limit": 1})
    if not test_data or "results" not in test_data:
        print("[API TEST FAILED] iTunes Search API unreachable.")
        sys.exit(1)
    print(f"[AUTH OK] iTunes Search API reachable.")

    crawler = iTunesCrawler(
        session=session,
        checkpoint=checkpoint,
        out_dir=out_dir,
        countries=countries,
        target_total=args.target_total,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        max_per_query=args.max_per_query,
        culture_override=args.culture_override,
        extra_terms=extra_terms,
    )

    crawler.run(resume=args.resume)


if __name__ == "__main__":
    main()
