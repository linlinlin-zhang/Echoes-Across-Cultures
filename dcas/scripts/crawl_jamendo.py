"""
Jamendo Bulk Crawler for DCAS

Distributed, resumable collection of Creative-Commons licensed tracks
from Jamendo with CultureMERT-compatible metadata output.

Usage:
    # Start fresh — global popular music sweep
    python -m dcas.scripts.crawl_jamendo \
        --client_id YOUR_JAMENDO_CLIENT_ID \
        --out_dir ./storage/public/jamendo_crawl \
        --target_total 20000 \
        --workers 6

    # Resume an interrupted crawl
    python -m dcas.scripts.crawl_jamendo \
        --client_id YOUR_JAMENDO_CLIENT_ID \
        --out_dir ./storage/public/jamendo_crawl \
        --resume

    # Culture-specific targeted crawl
    python -m dcas.scripts.crawl_jamendo \
        --client_id YOUR_JAMENDO_CLIENT_ID \
        --out_dir ./storage/public/jamendo_crawl_china \
        --cultures china,korea,japan \
        --target_total 5000
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


JAMENDO_API_BASE = "https://api.jamendo.com/v3.0"
JAMENDO_TRACKS_URL = f"{JAMENDO_API_BASE}/tracks"

PAGE_LIMIT = 200  # Jamendo max per page

# ---------------------------------------------------------------------------
# Culture-driven tag configurations
# ---------------------------------------------------------------------------

CULTURE_TAG_CONFIGS: dict[str, dict[str, Any]] = {
    "west": {
        "tags": [
            "pop", "rock", "hip-hop", "electronic", "indie", "folk",
            "jazz", "classical", "blues", "soul", "funk", "country",
            "metal", "punk", "reggae", "ambient", "lo-fi", "edm",
        ],
        "fuzzytags": "",
    },
    "china": {
        "tags": [
            "chinese", "mandopop", "cantopop", "c-pop", "chinese folk",
            "cantonese", "hakka", "hokkien", "taiwanese", "minnan",
            "teochew", "shanghainese",
        ],
        "fuzzytags": "chinese,china,mandarin,cantonese,hakka,hokkien,taiwanese,minnan,teochew,shanghainese",
    },
    "korea": {
        "tags": ["korean", "k-pop", "k-indie"],
        "fuzzytags": "korean,korea,kpop",
    },
    "japan": {
        "tags": ["japanese", "j-pop", "j-rock", "anime"],
        "fuzzytags": "japanese,japan,jpop,jrock",
    },
    "india": {
        "tags": ["indian", "bollywood", "carnatic", "hindustani", "bhangra"],
        "fuzzytags": "indian,india,bollywood",
    },
    "latin": {
        "tags": ["latin", "reggaeton", "salsa", "bachata", "cumbia", "tango"],
        "fuzzytags": "latin,reggaeton,salsa,bachata",
    },
    "brazil": {
        "tags": ["brazilian", "samba", "bossa nova", "mpb", "forro"],
        "fuzzytags": "brazilian,brazil,samba,bossa",
    },
    "africa": {
        "tags": ["african", "afrobeats", "highlife", "soukous", "amapiano"],
        "fuzzytags": "african,afrobeats,afrobeat,africa",
    },
    "middle_east": {
        "tags": ["arabic", "turkish", "persian", "oriental", "rai"],
        "fuzzytags": "arabic,turkish,persian,middle,east,oriental",
    },
    "southeast_asia": {
        "tags": ["thai", "vietnamese", "philippine", "malay", "indonesian", "dangdut"],
        "fuzzytags": "thai,vietnamese,philippine,malay,indonesian,dangdut",
    },
    "celtic": {
        "tags": ["celtic", "irish", "scottish", "breton", "galician", "gaelic", "fiddle"],
        "fuzzytags": "celtic,irish,scottish,gaelic,breton,galician",
    },
    "nordic": {
        "tags": ["nordic", "scandinavian", "swedish", "norwegian", "finnish", "danish", "icelandic"],
        "fuzzytags": "nordic,scandinavian,swedish,norwegian,finnish,danish,icelandic",
    },
    "eastern_europe": {
        "tags": ["polish", "ukrainian", "czech", "hungarian", "romanian", "slavic"],
        "fuzzytags": "polish,ukrainian,czech,hungarian,romanian,slavic,eastern,europe",
    },
    "balkans": {
        "tags": ["balkan", "greek", "serbian", "croatian", "bulgarian", "sevdah"],
        "fuzzytags": "balkan,greek,serbian,croatian,bulgarian,sevdah",
    },
    "caribbean": {
        "tags": ["caribbean", "reggae", "dancehall", "soca", "calypso", "zouk"],
        "fuzzytags": "caribbean,reggae,dancehall,soca,calypso,zouk,kompa",
    },
    "andean": {
        "tags": ["andean", "huayno", "quechua", "charango", "peruvian", "bolivian"],
        "fuzzytags": "andean,huayno,quechua,charango,peruvian,bolivian",
    },
    "central_asia": {
        "tags": ["kazakh", "uzbek", "kyrgyz", "tajik", "turkmen", "central asian"],
        "fuzzytags": "kazakh,uzbek,kyrgyz,tajik,turkmen,central,asian",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JamendoTrackRecord:
    track_id: str
    title: str
    artist: str
    album: str
    duration_ms: int
    culture: str
    label: str  # primary tag used for search
    tags: str   # all Jamendo tags joined
    jamendo_id: str
    jamendo_url: str
    audio_url: str
    audiodownload_url: str
    image_url: str | None = None
    license_url: str | None = None
    position: int = 0  # chart position if available

    def to_metadata_row(self, audio_rel_path: str) -> dict[str, str]:
        return {
            "track_id": self.track_id,
            "culture": self.culture,
            "audio_path": audio_rel_path,
            "source_dataset": "jamendo",
            "label": self.label,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "duration_ms": str(self.duration_ms),
            "tags": self.tags,
            "jamendo_id": self.jamendo_id,
            "jamendo_url": self.jamendo_url,
            "audio_url": self.audio_url,
            "image_url": self.image_url or "",
            "license_url": self.license_url or "",
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
# HTTP helpers
# ---------------------------------------------------------------------------

class RateLimitedSession:
    def __init__(self, max_retries: int = 5, default_backoff: float = 3.0):
        self.max_retries = max_retries
        self.default_backoff = default_backoff

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    sleep_secs = self.default_backoff * (2 ** attempt)
                    print(f"[RATE LIMIT] 429; sleeping {sleep_secs:.1f}s")
                    time.sleep(sleep_secs)
                    continue
                if resp.status_code in (502, 503, 504):
                    sleep_secs = self.default_backoff * (2 ** attempt)
                    print(f"[SERVER ERROR] {resp.status_code}; sleeping {sleep_secs:.1f}s")
                    time.sleep(sleep_secs)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                print(f"[REQUEST ERROR] {e} (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(self.default_backoff * (2 ** attempt))
        return {}


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

class JamendoCrawler:
    def __init__(
        self,
        client_id: str,
        session: RateLimitedSession,
        checkpoint: CheckpointManager,
        out_dir: Path,
        cultures: list[str],
        target_total: int,
        workers: int,
        checkpoint_interval: int,
        max_per_query: int,
    ):
        self.client_id = client_id
        self.session = session
        self.checkpoint = checkpoint
        self.out_dir = out_dir
        self.cultures = cultures
        self.target_total = target_total
        self.workers = workers
        self.checkpoint_interval = checkpoint_interval
        self.max_per_query = max(1, int(max_per_query))

        self.audio_dir = out_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._fieldnames = [
            "track_id", "culture", "audio_path", "source_dataset", "label",
            "title", "artist", "album", "duration_ms",
            "tags", "jamendo_id", "jamendo_url", "audio_url",
            "image_url", "license_url",
        ]

    def _fetch_page(
        self,
        tags: str = "",
        fuzzytags: str = "",
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "client_id": self.client_id,
            "format": "json",
            "limit": PAGE_LIMIT,
            "offset": offset,
            "include": "musicinfo+stats+lyrics",
            "audioformat": "mp32",
            "audiodlformat": "mp32",
        }
        if tags:
            params["tags"] = tags
        if fuzzytags:
            params["fuzzytags"] = fuzzytags

        data = self.session.get(JAMENDO_TRACKS_URL, params=params)
        if not data or "results" not in data:
            return []
        headers = data.get("headers") or {}
        if str(headers.get("status", "")).lower() == "failed":
            print(f"[API ERROR] Jamendo returned failed status: {headers.get('error_message', 'unknown error')}")
            return []
        return list(data["results"])

    @staticmethod
    def _parse_track(
        item: dict[str, Any],
        culture: str,
        label: str,
    ) -> JamendoTrackRecord | None:
        tid = str(item.get("id", ""))
        if not tid:
            return None
        tags_list = item.get("tags", []) or []
        if isinstance(tags_list, str):
            tags_list = [t.strip() for t in tags_list.split(",")]
        tags_str = ",".join(str(t) for t in tags_list)

        # Prefer audiodownload URL if available (higher quality, CC-licensed)
        audio_url = item.get("audiodownload") or item.get("audio") or ""
        if not audio_url:
            return None

        return JamendoTrackRecord(
            track_id=f"jamendo_{tid}",
            title=str(item.get("name", "")),
            artist=str(item.get("artist_name", "Unknown")),
            album=str(item.get("album_name", "")),
            duration_ms=int(item.get("duration", 0)) * 1000,
            culture=culture,
            label=label,
            tags=tags_str,
            jamendo_id=tid,
            jamendo_url=item.get("shareurl", "") or f"https://www.jamendo.com/track/{tid}",
            audio_url=audio_url,
            audiodownload_url=audio_url,
            image_url=item.get("album_image", ""),
            license_url=item.get("license_ccurl", ""),
        )

    def download_track(self, record: JamendoTrackRecord) -> Path | None:
        safe_tid = re.sub(r"[^a-zA-Z0-9._-]", "_", record.track_id)
        safe_culture = re.sub(r"[^a-zA-Z0-9._-]", "_", record.culture)[:32]
        safe_label = re.sub(r"[^a-zA-Z0-9._-]", "_", record.label)[:32]
        subdir = self.audio_dir / safe_culture / safe_label
        subdir.mkdir(parents=True, exist_ok=True)
        dest = subdir / f"{safe_tid}.mp3"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest
        try:
            resp = requests.get(record.audio_url, timeout=60, stream=True)
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
        """Return list of (culture, tags, fuzzytags) query specs."""
        pool: list[tuple[str, str, str]] = []
        for culture in self.cultures:
            cfg = CULTURE_TAG_CONFIGS.get(culture, {})
            tags_list = cfg.get("tags", [])
            fuzzytags = cfg.get("fuzzytags", "")
            # For each explicit tag, create a dedicated query
            for tag in tags_list:
                pool.append((culture, tag, ""))
            # Also add a broad fuzzy query if available
            if fuzzytags:
                pool.append((culture, "", fuzzytags))
        random.shuffle(pool)
        return pool

    def run(self, resume: bool = False) -> dict[str, Any]:
        state = CrawlState()
        if resume and self.checkpoint.exists():
            state = self.checkpoint.load()
            print(f"[RESUME] collected={state.total_collected}, downloaded={state.total_downloaded}")
        else:
            if self.metadata_path.exists() and not resume:
                print("[WARN] metadata.csv exists but --resume not set. Overwriting on start.")

        completed_set = set(state.completed_queries)
        downloaded_set = set(state.downloaded_track_ids)
        failed_set = set(state.failed_track_ids)
        existing_meta_ids = self.checkpoint.read_existing_metadata_ids()
        downloaded_set |= existing_meta_ids
        if resume and state.completed_queries and not downloaded_set and not failed_set:
            print("[RECOVERY] Existing checkpoint has completed queries but no downloaded/failed tracks; retrying queries.")
            state.completed_queries.clear()
            state.total_collected = 0
            completed_set.clear()
        seen_set = set(downloaded_set) | set(failed_set)
        print(f"[INIT] {len(downloaded_set)} tracks already in metadata.csv")

        query_pool = self._build_query_pool()
        total_queries = len(query_pool)
        print(f"[INIT] Query pool: {total_queries} queries across cultures={self.cultures}")

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

        def process_download_batch(records: list[JamendoTrackRecord]) -> None:
            nonlocal pending_metadata_rows
            to_download = [r for r in records if r.track_id not in downloaded_set and r.track_id not in failed_set]
            if not to_download:
                return
            completed_in_batch = 0
            flush_every = max(10, self.workers * 2)
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {ex.submit(self.download_track, r): r for r in to_download}
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
                    completed_in_batch += 1
                    if completed_in_batch % flush_every == 0:
                        print(
                            f"[DOWNLOAD PROGRESS] batch={completed_in_batch}/{len(to_download)} "
                            f"total_downloaded={state.total_downloaded} failed={len(state.failed_track_ids)}"
                        )
                        do_checkpoint(force=True)

        try:
            for qidx, (culture, tags, fuzzytags) in enumerate(query_pool, start=1):
                query_key = f"{culture}::{tags}::{fuzzytags}"
                if query_key in completed_set:
                    continue

                if state.total_collected >= self.target_total:
                    print(f"[TARGET REACHED] {state.total_collected} >= {self.target_total}")
                    break

                print(f"[{qidx}/{total_queries}] culture={culture} tags={tags or '(fuzzy)'} fuzzytags={fuzzytags or '(none)'}")

                all_records: list[JamendoTrackRecord] = []
                remaining = max(0, self.target_total - state.total_collected)
                query_limit = min(remaining, self.max_per_query)
                for offset in range(0, 10_000, PAGE_LIMIT):
                    items = self._fetch_page(tags=tags, fuzzytags=fuzzytags, offset=offset)
                    if not items:
                        break
                    for item in items:
                        rec = self._parse_track(item, culture, tags or fuzzytags)
                        if rec and rec.track_id not in seen_set:
                            all_records.append(rec)
                            seen_set.add(rec.track_id)
                            if len(all_records) >= query_limit:
                                break
                    if len(all_records) >= query_limit:
                        break
                    if len(items) < PAGE_LIMIT:
                        break
                    time.sleep(0.3)
                    if state.total_collected + len(all_records) >= self.target_total:
                        break

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
                time.sleep(0.5)

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
    ap = argparse.ArgumentParser(description="Bulk collect Jamendo CC-licensed tracks for DCAS.")
    ap.add_argument("--client_id", default=os.environ.get("JAMENDO_CLIENT_ID", ""), help="Jamendo API Client ID (or set JAMENDO_CLIENT_ID)")
    ap.add_argument("--out_dir", required=True, help="Output directory for audio + metadata")
    ap.add_argument("--cultures", default="", help="Comma-separated cultures to crawl (default: all)")
    ap.add_argument("--target_total", type=int, default=20_000, help="Target unique tracks")
    ap.add_argument("--workers", type=int, default=6, help="Parallel download workers")
    ap.add_argument("--checkpoint_interval", type=int, default=300, help="Seconds between checkpoints")
    ap.add_argument("--max_per_query", type=int, default=50, help="Max new tracks downloaded from a single culture/tag query")
    ap.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not str(args.client_id).strip():
        print("[AUTH/API TEST FAILED] Missing Jamendo Client ID. Set --client_id or JAMENDO_CLIENT_ID.")
        sys.exit(1)

    cultures = [c.strip().lower() for c in args.cultures.split(",") if c.strip()]
    if not cultures:
        cultures = list(CULTURE_TAG_CONFIGS.keys())
    else:
        # Validate
        invalid = [c for c in cultures if c not in CULTURE_TAG_CONFIGS]
        if invalid:
            print(f"[ERROR] Unknown cultures: {invalid}. Valid: {list(CULTURE_TAG_CONFIGS.keys())}")
            sys.exit(1)

    session = RateLimitedSession()
    checkpoint = CheckpointManager(out_dir)

    # Quick API test
    test_data = session.get(
        JAMENDO_TRACKS_URL,
        params={
            "client_id": args.client_id,
            "format": "json",
            "limit": 1,
        },
    )
    test_headers = (test_data or {}).get("headers") or {}
    if (
        not test_data
        or "results" not in test_data
        or str(test_headers.get("status", "")).lower() == "failed"
    ):
        if test_headers.get("error_message"):
            print(f"[AUTH/API TEST FAILED] {test_headers.get('error_message')}")
        else:
            print("[AUTH/API TEST FAILED] Check your Jamendo Client ID.")
        sys.exit(1)
    print(f"[AUTH OK] Jamendo API reachable.")

    crawler = JamendoCrawler(
        client_id=args.client_id,
        session=session,
        checkpoint=checkpoint,
        out_dir=out_dir,
        cultures=cultures,
        target_total=args.target_total,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        max_per_query=args.max_per_query,
    )

    crawler.run(resume=args.resume)


if __name__ == "__main__":
    main()
