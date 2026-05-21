"""
Spotify Preview Crawler for DCAS

Distributed, resumable bulk collection of Spotify track previews with
CultureMERT-compatible metadata output.

Features:
- Multi-market, multi-genre search strategies to maximize coverage
- Checkpoint/resume for long-running interrupted jobs
- Automatic rate-limit backoff (429 handling)
- Parallel preview MP3 downloads
- Standard metadata.csv output compatible with build_tracks_from_audio.py

Usage:
    # Start a fresh crawl targeting 50k tracks
    python -m dcas.scripts.crawl_spotify_previews \
        --client_id YOUR_ID --client_secret YOUR_SECRET \
        --out_dir ./storage/public/spotify_crawl \
        --target_total 50000 \
        --workers 4

    # Resume an interrupted crawl
    python -m dcas.scripts.crawl_spotify_previews \
        --client_id YOUR_ID --client_secret YOUR_SECRET \
        --out_dir ./storage/public/spotify_crawl \
        --resume

    # Fast mode: skip audio features, only previews
    python -m dcas.scripts.crawl_spotify_previews \
        --client_id YOUR_ID --client_secret YOUR_SECRET \
        --out_dir ./storage/public/spotify_crawl \
        --skip_audio_features \
        --workers 8
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPOTIFY_ACCOUNTS_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

SEARCH_LIMIT = 10  # Development Mode caps search limit at 10
MAX_SEARCH_OFFSET = 1000
AUDIO_FEATURES_BATCH = 100

DEFAULT_MARKETS = [
    # Americas
    "US", "BR", "MX", "CA", "AR", "CO", "CL",
    # Europe
    "GB", "DE", "FR", "ES", "NL", "SE", "IT", "PL", "TR", "RU",
    # Asia-Pacific
    "JP", "KR", "IN", "ID", "TH", "PH", "MY", "AU", "NZ", "TW", "HK", "SG",
    # Africa / Middle East
    "ZA", "NG", "EG", "IL", "SA",
]

GENRE_QUERIES_BY_CULTURE: dict[str, list[str]] = {
    "west": [
        "pop", "rock", "hip-hop", "electronic", "country", "folk",
        "jazz", "classical", "blues", "r&b", "indie pop", "indie rock",
        "alternative rock", "punk", "metal", "soul", "funk", "disco",
        "reggae", "ska", "gospel", "new wave", "synth-pop", "edm",
        "techno", "house", "trance", "dubstep", "ambient", "lo-fi",
        "progressive rock", "hard rock", "soft rock", "pop rock",
    ],
    "korea": [
        "k-pop", "k-indie", "k-rap", "k-r&b", "korean ballad", "trot",
    ],
    "japan": [
        "j-pop", "j-rock", "j-metal", "city pop", "anime", "enka",
        "visual kei", "shibuya-kei", "j-hip-hop", "jazz japan",
    ],
    "india": [
        "bollywood", "indian pop", "indian classical", "carnatic",
        "hindustani", "bhangra", "punjabi", "telugu", "tamil pop",
        "malayalam", "ghazal", "sufi", "devotional", "indie india",
    ],
    "china": [
        "mandopop", "cantopop", "c-pop", "chinese indie", "chinese rock",
        "chinese folk", "chinese classical", "taiwan indie", "taiwan pop",
    ],
    "latin": [
        "reggaeton", "latin pop", "salsa", "bachata", "cumbia", "tango",
        "merengue", "dembow", "latin rock", "latin hip-hop", "mariachi",
        "ranchera", "tejano", "vallenato", "bolero", "flamenco",
    ],
    "brazil": [
        "samba", "bossa nova", "mpb", "sertanejo", "funk carioca",
        "forro", "axé", "pagode", "brazilian rock", "brazilian hip-hop",
        "tropicália", "choro",
    ],
    "africa": [
        "afrobeats", "amapiano", "highlife", "soukous", "mbalax",
        "gnawa", "rai", "afrobeat", "afro-fusion", "afro-pop",
        "benga", "bongo flava", "coupé-décalé", "kizomba", "zouk",
    ],
    "middle_east": [
        "arabic pop", "turkish pop", "turkish rock", "persian classical",
        "iranian pop", "levantine", "dabke", "tarab", "mugham",
        "ottoman classical", "kurdish", "assyrian pop",
    ],
    "southeast_asia": [
        "thai pop", "thai rock", "phleng phuea chiwit", "luk thung",
        "vietnamese pop", "v-pop", "pinoy pop", "p-pop", "opm",
        "malay pop", "dangdut", "indonesian indie", "khmer pop",
    ],
}

ALL_QUERIES: list[tuple[str, str, str]] = []
for _culture, _genres in GENRE_QUERIES_BY_CULTURE.items():
    for _genre in _genres:
        ALL_QUERIES.append((_culture, _genre))

# Years to rotate through for extra coverage
YEAR_RANGES = [
    "1950-1970", "1970-1980", "1980-1990", "1990-2000",
    "2000-2005", "2005-2010", "2010-2015", "2015-2018",
    "2018-2020", "2020-2022", "2022-2024", "2024-2026",
]

# Market -> inferred culture for market_year search mode
MARKET_TO_CULTURE: dict[str, str] = {
    "US": "west", "GB": "west", "CA": "west", "AU": "west", "NZ": "west",
    "DE": "west", "FR": "west", "ES": "west", "NL": "west", "SE": "west",
    "IT": "west", "PL": "west", "IE": "west", "NO": "west", "FI": "west",
    "DK": "west", "AT": "west", "CH": "west", "BE": "west", "PT": "west",
    "JP": "japan", "KR": "korea",
    "IN": "india",
    "TW": "china", "HK": "china", "SG": "china", "MO": "china",
    "BR": "brazil",
    "MX": "latin", "CO": "latin", "CL": "latin", "AR": "latin", "PE": "latin",
    "VE": "latin", "EC": "latin", "UY": "latin", "PY": "latin", "BO": "latin",
    "ZA": "africa", "NG": "africa", "EG": "africa", "GH": "africa", "KE": "africa",
    "TZ": "africa", "UG": "africa", "MZ": "africa", "ZM": "africa", "ZW": "africa",
    "TR": "middle_east", "IL": "middle_east", "SA": "middle_east", "AE": "middle_east",
    "QA": "middle_east", "KW": "middle_east", "BH": "middle_east", "OM": "middle_east",
    "JO": "middle_east", "LB": "middle_east", "IQ": "middle_east", "IR": "middle_east",
    "RU": "middle_east",  # Approximate for music cultural sphere
    "ID": "southeast_asia", "TH": "southeast_asia", "PH": "southeast_asia",
    "MY": "southeast_asia", "VN": "southeast_asia", "KH": "southeast_asia",
    "LA": "southeast_asia", "MM": "southeast_asia", "BN": "southeast_asia",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackRecord:
    track_id: str
    title: str
    artist: str
    album: str
    popularity: int
    preview_url: str | None
    culture: str
    label: str  # genre / search query
    market: str
    duration_ms: int = 0
    explicit: bool = False
    # Audio features (optional)
    danceability: float | None = None
    energy: float | None = None
    key: int | None = None
    loudness: float | None = None
    mode: int | None = None
    speechiness: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    liveness: float | None = None
    valence: float | None = None
    tempo: float | None = None
    time_signature: int | None = None

    def to_metadata_row(self, audio_rel_path: str) -> dict[str, str]:
        """Emit a row compatible with build_tracks_from_audio.py metadata.csv."""
        row: dict[str, str] = {
            "track_id": self.track_id,
            "culture": self.culture,
            "audio_path": audio_rel_path,
            "label": self.label,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "popularity": str(self.popularity),
            "market": self.market,
            "duration_ms": str(self.duration_ms),
            "explicit": str(self.explicit),
        }
        for af in (
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo", "time_signature",
        ):
            v = getattr(self, af)
            row[af] = f"{v:.6f}" if isinstance(v, float) else (str(v) if v is not None else "")
        if self.preview_url:
            row["preview_url"] = self.preview_url
        return row


@dataclass
class CrawlState:
    version: int = 1
    completed_queries: list[str] = field(default_factory=list)
    downloaded_track_ids: list[str] = field(default_factory=list)
    failed_track_ids: list[str] = field(default_factory=list)
    total_collected: int = 0
    total_with_preview: int = 0
    total_downloaded: int = 0
    total_audio_features_fetched: int = 0

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
            total_with_preview=int(d.get("total_with_preview", 0)),
            total_downloaded=int(d.get("total_downloaded", 0)),
            total_audio_features_fetched=int(d.get("total_audio_features_fetched", 0)),
        )


# ---------------------------------------------------------------------------
# Auth & Rate-Limited HTTP
# ---------------------------------------------------------------------------

class SpotifyAuth:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str | None = None
        self._expires_at: float = 0.0

    def _fetch_token(self) -> str:
        creds = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        resp = requests.post(
            SPOTIFY_ACCOUNTS_URL,
            headers={"Authorization": f"Basic {creds}"},
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = str(data["access_token"])
        # Refresh 60 seconds early
        self._expires_at = time.time() + float(data.get("expires_in", 3600)) - 60
        return self._token

    def get_token(self) -> str:
        if self._token is None or time.time() >= self._expires_at:
            return self._fetch_token()
        return self._token


class RateLimitedSession:
    """Wraps requests with automatic 429 backoff."""

    def __init__(self, max_retries: int = 5, default_backoff: float = 5.0):
        self.max_retries = max_retries
        self.default_backoff = default_backoff
        self._consecutive_429 = 0

    def request(self, method: str, url: str, headers: dict[str, str] | None = None, **kwargs: Any) -> requests.Response:
        headers = dict(headers) if headers else {}
        for attempt in range(self.max_retries):
            resp = requests.request(method, url, headers=headers, timeout=30, **kwargs)
            if resp.status_code == 429:
                self._consecutive_429 += 1
                # Spotify returns Retry-After in seconds
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    sleep_secs = float(retry_after) + 1.0
                else:
                    sleep_secs = self.default_backoff * (2 ** self._consecutive_429)
                sleep_secs = min(sleep_secs, 300.0)
                print(f"[RATE LIMIT] 429 on {url}; sleeping {sleep_secs:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(sleep_secs)
                continue
            if resp.status_code in (502, 503, 504):
                sleep_secs = self.default_backoff * (2 ** attempt)
                print(f"[SERVER ERROR] {resp.status_code} on {url}; sleeping {sleep_secs:.1f}s")
                time.sleep(sleep_secs)
                continue
            self._consecutive_429 = max(0, self._consecutive_429 - 1)
            return resp
        resp.raise_for_status()
        return resp

    def get(self, url: str, headers: dict[str, str] | None = None, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, headers=headers, **kwargs)


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.state_path = out_dir / "state.json"
        self.metadata_path = out_dir / "metadata.csv"
        self._lock_file = out_dir / ".state_lock"

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
# Crawler
# ---------------------------------------------------------------------------

class SpotifyCrawler:
    def __init__(
        self,
        auth: SpotifyAuth,
        session: RateLimitedSession,
        checkpoint: CheckpointManager,
        out_dir: Path,
        markets: list[str],
        skip_audio_features: bool,
        target_total: int,
        workers: int,
        checkpoint_interval: int,
        shuffle_queries: bool,
        search_mode: str = "genre",
    ):
        self.auth = auth
        self.session = session
        self.checkpoint = checkpoint
        self.out_dir = out_dir
        self.markets = markets
        self.skip_audio_features = skip_audio_features
        self.target_total = target_total
        self.workers = workers
        self.checkpoint_interval = checkpoint_interval
        self.shuffle_queries = shuffle_queries
        self.search_mode = search_mode

        self.audio_dir = out_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._fieldnames = [
            "track_id", "culture", "audio_path", "label",
            "title", "artist", "album", "popularity", "market",
            "duration_ms", "explicit",
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo", "time_signature",
            "preview_url",
        ]

    def _api_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.auth.get_token()}"}

    def search_tracks(self, market: str, query: str, offset: int = 0) -> list[dict[str, Any]]:
        url = f"{SPOTIFY_API_BASE}/search"
        params = {
            "q": query,
            "type": "track",
            "market": market,
            "limit": SEARCH_LIMIT,
            "offset": offset,
        }
        resp = self.session.get(url, headers=self._api_headers(), params=params)
        if resp.status_code != 200:
            print(f"[SEARCH ERROR] {resp.status_code} for q={query} market={market} offset={offset}")
            return []
        data = resp.json()
        items = data.get("tracks", {}).get("items", [])
        return items

    def fetch_audio_features(self, track_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not track_ids or self.skip_audio_features:
            return {}
        result: dict[str, dict[str, Any]] = {}
        # Batch in 100s
        for i in range(0, len(track_ids), AUDIO_FEATURES_BATCH):
            batch = track_ids[i : i + AUDIO_FEATURES_BATCH]
            ids_param = ",".join(batch)
            url = f"{SPOTIFY_API_BASE}/audio-features"
            resp = self.session.get(url, headers=self._api_headers(), params={"ids": ids_param})
            if resp.status_code != 200:
                print(f"[AUDIO_FEATURES ERROR] {resp.status_code} for batch {i}")
                continue
            data = resp.json()
            for feat in data.get("audio_features", []) or []:
                if feat and feat.get("id"):
                    result[feat["id"]] = feat
        return result

    @staticmethod
    def _parse_track(item: dict[str, Any], culture: str, label: str, market: str) -> TrackRecord:
        artists = item.get("artists", [])
        artist_name = artists[0]["name"] if artists else "Unknown"
        album_name = item.get("album", {}).get("name", "")
        return TrackRecord(
            track_id=str(item["id"]),
            title=str(item.get("name", "")),
            artist=artist_name,
            album=album_name,
            popularity=int(item.get("popularity", 0)),
            preview_url=item.get("preview_url"),
            culture=culture,
            label=label,
            market=market,
            duration_ms=int(item.get("duration_ms", 0)),
            explicit=bool(item.get("explicit", False)),
        )

    def download_preview(self, record: TrackRecord) -> Path | None:
        if not record.preview_url:
            return None
        safe_tid = re.sub(r"[^a-zA-Z0-9._-]", "_", record.track_id)
        # Organize by culture/genre for easier inspection
        safe_culture = re.sub(r"[^a-zA-Z0-9._-]", "_", record.culture)[:32]
        safe_label = re.sub(r"[^a-zA-Z0-9._-]", "_", record.label)[:32]
        subdir = self.audio_dir / safe_culture / safe_label
        subdir.mkdir(parents=True, exist_ok=True)
        dest = subdir / f"{safe_tid}.mp3"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest
        try:
            resp = requests.get(record.preview_url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
            if dest.stat().st_size < 1024:
                dest.unlink()
                return None
            return dest
        except Exception as e:
            print(f"[DOWNLOAD ERROR] {record.track_id}: {e}")
            if dest.exists():
                dest.unlink(missing_ok=True)
            return None

    def _build_genre_query_pool(self) -> list[tuple[str, str, str]]:
        """Genre-based search across markets."""
        pool: list[tuple[str, str, str]] = []
        for market in self.markets:
            for culture, genres in GENRE_QUERIES_BY_CULTURE.items():
                for genre in genres:
                    base_q = f"genre:{genre}"
                    pool.append((market, base_q, f"{culture}:{genre}"))
                    if random.random() < 0.4:
                        year_range = random.choice(YEAR_RANGES)
                        year_q = f"genre:{genre} year:{year_range}"
                        pool.append((market, year_q, f"{culture}:{genre}"))
        return pool

    def _build_market_year_query_pool(self) -> list[tuple[str, str, str]]:
        """Market+year sweep: captures each market's popular music by decade/year."""
        pool: list[tuple[str, str, str]] = []
        # Primary sweep: year-by-year from 2026 back to 1950
        years = list(range(2026, 1949, -1))
        # Secondary: recent years get more granular attention
        recent_years = list(range(2026, 2019, -1))
        older_decades = list(range(2010, 1949, -10))

        for market in self.markets:
            culture = MARKET_TO_CULTURE.get(market, "west")
            # Every single year (primary)
            for year in years:
                q = f"year:{year}"
                pool.append((market, q, f"{culture}:year_{year}"))
            # Recent years get a second pass with popularity bias hint
            for year in recent_years:
                q = f"year:{year} pop"
                pool.append((market, q, f"{culture}:pop_{year}"))
            # Decade-wide searches for older eras
            for decade_start in older_decades:
                decade_end = decade_start + 9
                q = f"year:{decade_start}-{decade_end}"
                pool.append((market, q, f"{culture}:decade_{decade_start}s"))
        return pool

    def _build_query_pool(self) -> list[tuple[str, str, str]]:
        """Return list of (market, query_string, culture_label) tuples."""
        if self.search_mode == "market_year":
            pool = self._build_market_year_query_pool()
            print(f"[QUERY POOL] market_year mode: {len(pool)} queries")
        else:
            pool = self._build_genre_query_pool()
            print(f"[QUERY POOL] genre mode: {len(pool)} queries")
        if self.shuffle_queries:
            random.shuffle(pool)
        return pool

    def run(self, resume: bool = False) -> dict[str, Any]:
        state = CrawlState()
        if resume and self.checkpoint.exists():
            state = self.checkpoint.load()
            print(f"[RESUME] Loaded state: collected={state.total_collected}, with_preview={state.total_with_preview}, downloaded={state.total_downloaded}")
        else:
            if self.metadata_path.exists() and not resume:
                print("[WARN] metadata.csv already exists but --resume not set. Starting fresh will overwrite state.")

        completed_set = set(state.completed_queries)
        downloaded_set = set(state.downloaded_track_ids)
        failed_set = set(state.failed_track_ids)

        # Also read any IDs already in metadata.csv that might not be in state
        existing_meta_ids = self.checkpoint.read_existing_metadata_ids()
        downloaded_set |= existing_meta_ids
        print(f"[INIT] {len(downloaded_set)} tracks already in metadata.csv")

        query_pool = self._build_query_pool()
        total_queries = len(query_pool)
        print(f"[INIT] Query pool size: {total_queries} (markets={len(self.markets)})")

        last_checkpoint_time = time.time()
        pending_records: list[TrackRecord] = []
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
            print(f"[CHECKPOINT] saved state: collected={state.total_collected}, with_preview={state.total_with_preview}, downloaded={state.total_downloaded}")

        def process_download_batch(records: list[TrackRecord]) -> None:
            """Download previews in parallel and emit metadata rows."""
            nonlocal pending_metadata_rows
            to_download = [r for r in records if r.preview_url and r.track_id not in downloaded_set and r.track_id not in failed_set]
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

        # ------------------------------------------------------------------
        # Main crawl loop
        # ------------------------------------------------------------------
        try:
            for qidx, (market, query, culture_label) in enumerate(query_pool, start=1):
                query_key = f"{market}::{query}"
                if query_key in completed_set:
                    continue

                if state.total_collected >= self.target_total:
                    print(f"[TARGET REACHED] collected={state.total_collected} >= target={self.target_total}")
                    break

                print(f"[{qidx}/{total_queries}] market={market} query={query}")

                all_items: list[dict[str, Any]] = []
                for offset in range(0, MAX_SEARCH_OFFSET, SEARCH_LIMIT):
                    try:
                        items = self.search_tracks(market, query, offset=offset)
                    except Exception as e:
                        print(f"[SEARCH EXCEPTION] {e}")
                        break
                    if not items:
                        break
                    all_items.extend(items)
                    # Small sleep to be polite
                    time.sleep(0.1)
                    if state.total_collected + len(all_items) >= self.target_total:
                        break

                if not all_items:
                    completed_set.add(query_key)
                    state.completed_queries.append(query_key)
                    do_checkpoint()
                    continue

                # Parse and dedupe against global set
                culture, _sep, label = culture_label.partition(":")
                records: list[TrackRecord] = []
                for item in all_items:
                    tid = item.get("id")
                    if not tid or tid in downloaded_set or tid in failed_set:
                        continue
                    rec = self._parse_track(item, culture, label, market)
                    records.append(rec)
                    downloaded_set.add(tid)  # mark as seen so we don't re-add from another query

                state.total_collected += len(records)
                with_preview = [r for r in records if r.preview_url]
                state.total_with_preview += len(with_preview)
                print(f"  -> new records={len(records)}, with_preview={len(with_preview)}")

                # Fetch audio features for this batch (optional)
                if not self.skip_audio_features and with_preview:
                    try:
                        af_map = self.fetch_audio_features([r.track_id for r in with_preview])
                        for r in with_preview:
                            af = af_map.get(r.track_id)
                            if af:
                                for k in (
                                    "danceability", "energy", "key", "loudness", "mode",
                                    "speechiness", "acousticness", "instrumentalness",
                                    "liveness", "valence", "tempo", "time_signature",
                                ):
                                    setattr(r, k, af.get(k))
                        state.total_audio_features_fetched += len(af_map)
                    except Exception as e:
                        print(f"[AUDIO_FEATURES EXCEPTION] {e}")

                # Download previews in parallel
                if with_preview:
                    process_download_batch(with_preview)

                completed_set.add(query_key)
                state.completed_queries.append(query_key)
                do_checkpoint()

                # Polite delay between queries
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Saving checkpoint before exit...")
        finally:
            do_checkpoint(force=True)

        # Final summary
        report = {
            "out_dir": str(self.out_dir.resolve()),
            "target_total": self.target_total,
            "total_collected": state.total_collected,
            "total_with_preview": state.total_with_preview,
            "total_downloaded": state.total_downloaded,
            "total_audio_features_fetched": state.total_audio_features_fetched,
            "completed_queries": len(state.completed_queries),
            "failed_tracks": len(state.failed_track_ids),
            "metadata_csv": str(self.checkpoint.metadata_path.resolve()),
        }
        report_path = self.out_dir / "import_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[DONE] Report saved to {report_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    @property
    def metadata_path(self) -> Path:
        return self.checkpoint.metadata_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk collect Spotify track previews for DCAS.")
    ap.add_argument("--client_id", required=True, help="Spotify App Client ID")
    ap.add_argument("--client_secret", required=True, help="Spotify App Client Secret")
    ap.add_argument("--out_dir", required=True, help="Output directory for audio + metadata")
    ap.add_argument("--markets", default="", help="Comma-separated market codes (default: global set of ~30)")
    ap.add_argument("--target_total", type=int, default=100_000, help="Target number of unique tracks to collect")
    ap.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    ap.add_argument("--checkpoint_interval", type=int, default=300, help="Seconds between checkpoint writes")
    ap.add_argument("--skip_audio_features", action="store_true", help="Skip fetching Spotify audio-features (faster)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing checkpoint/state.json")
    ap.add_argument("--no_shuffle", action="store_true", help="Do not shuffle query order (default: shuffle)")
    ap.add_argument(
        "--search_mode",
        choices=["genre", "market_year"],
        default="genre",
        help=(
            "Search strategy: 'genre' scans by genre across markets; "
            "'market_year' sweeps each market by year to capture local popular music."
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    if not markets:
        markets = list(DEFAULT_MARKETS)

    auth = SpotifyAuth(args.client_id, args.client_secret)
    session = RateLimitedSession()
    checkpoint = CheckpointManager(out_dir)

    # Quick auth test
    try:
        token = auth.get_token()
        print(f"[AUTH OK] Token prefix: {token[:8]}...")
    except Exception as e:
        print(f"[AUTH FAILED] {e}")
        sys.exit(1)

    crawler = SpotifyCrawler(
        auth=auth,
        session=session,
        checkpoint=checkpoint,
        out_dir=out_dir,
        markets=markets,
        skip_audio_features=args.skip_audio_features,
        target_total=args.target_total,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        shuffle_queries=not args.no_shuffle,
        search_mode=args.search_mode,
    )

    crawler.run(resume=args.resume)


if __name__ == "__main__":
    main()
