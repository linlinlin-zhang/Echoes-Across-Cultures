from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests


JAMENDO_TRACKS_URL = "https://api.jamendo.com/v3.0/tracks"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _release_year(value: Any) -> str:
    match = re.search(r"(?:19|20)\d{2}", _clean(value))
    return match.group(0) if match else ""


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _ensure_fields(fieldnames: list[str], fields: list[str]) -> list[str]:
    out = list(fieldnames)
    for field in fields:
        if field not in out:
            out.append(field)
    return out


def _load_client_id() -> str:
    value = _clean(os.environ.get("JAMENDO_CLIENT_ID"))
    if value:
        return value
    run_script = Path("run_jamendo_crawl.ps1")
    if not run_script.exists():
        return ""
    match = re.search(r"\$JAMENDO_CLIENT_ID\s*=\s*\"([^\"]+)\"", run_script.read_text(encoding="utf-8", errors="ignore"))
    if not match:
        return ""
    value = _clean(match.group(1))
    if value.startswith("<") or value.endswith(">"):
        return ""
    return value


def _jamendo_id(row: dict[str, str]) -> str:
    value = _clean(row.get("jamendo_id"))
    if value:
        return value
    track_id = _clean(row.get("track_id"))
    if track_id.startswith("jamendo_"):
        return track_id.removeprefix("jamendo_")
    return ""


def _album_id_from_url(value: str) -> str:
    if not _clean(value):
        return ""
    try:
        query = parse_qs(urlparse(value).query)
    except Exception:
        return ""
    ids = query.get("id") or query.get("album_id")
    return _clean(ids[0]) if ids else ""


def _album_id(row: dict[str, str]) -> str:
    return (
        _clean(row.get("album_id"))
        or _album_id_from_url(_clean(row.get("image_url")))
        or _album_id_from_url(_clean(row.get("cover_art_url")))
        or _album_id_from_url(_clean(row.get("cover_art_url_large")))
    )


def _is_jamendo(row: dict[str, str]) -> bool:
    return _clean(row.get("source_dataset")).lower() == "jamendo" or _clean(row.get("track_id")).startswith("jamendo_")


def _request_json(session: requests.Session, params: dict[str, Any], *, max_retries: int = 5) -> dict[str, Any]:
    for attempt in range(max_retries):
        response = session.get(JAMENDO_TRACKS_URL, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(min(45.0, 2.5 * (2 ** attempt)))
            continue
        response.raise_for_status()
        data = response.json()
        headers = data.get("headers") or {}
        if str(headers.get("status", "")).lower() == "failed":
            return {}
        return data
    response.raise_for_status()
    return {}


def _record_from_item(item: dict[str, Any]) -> dict[str, str]:
    release_date = _clean(item.get("releasedate"))
    album_id = _clean(item.get("album_id"))
    return {
        "jamendo_id": _clean(item.get("id")),
        "release_date": release_date,
        "release_year": _release_year(release_date),
        "album_id": album_id,
    }


def _fetch_album_batch(
    session: requests.Session,
    *,
    client_id: str,
    album_ids: list[str],
    delay_seconds: float,
) -> tuple[dict[str, dict[str, str]], int]:
    records: dict[str, dict[str, str]] = {}
    requests_made = 0
    offset = 0
    while True:
        params: list[tuple[str, Any]] = [
            ("client_id", client_id),
            ("format", "json"),
            ("limit", 200),
            ("offset", offset),
            ("include", "musicinfo+stats+lyrics"),
        ]
        params.extend(("album_id[]", album_id) for album_id in album_ids)
        data = _request_json(session, params)
        requests_made += 1
        for item in data.get("results") or []:
            record = _record_from_item(item)
            if record["jamendo_id"] and record["release_date"]:
                records[record["jamendo_id"]] = record
        headers = data.get("headers") or {}
        result_count = int(headers.get("results_count") or len(data.get("results") or []))
        offset += 200
        if result_count < 200:
            break
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return records, requests_made


def _fetch_track(
    session: requests.Session,
    *,
    client_id: str,
    jamendo_id: str,
) -> tuple[dict[str, str], int]:
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": 1,
        "id": jamendo_id,
        "include": "musicinfo+stats+lyrics",
    }
    data = _request_json(session, params)
    results = data.get("results") or []
    if not results:
        return {}, 1
    return _record_from_item(results[0]), 1


def enrich_files(
    paths: list[Path],
    *,
    client_id: str,
    album_batch_size: int,
    delay_seconds: float,
) -> dict[str, Any]:
    loaded: list[tuple[Path, list[dict[str, str]], list[str]]] = []
    album_ids: set[str] = set()
    fallback_ids: set[str] = set()
    missing_before = 0
    jamendo_rows = 0

    for path in paths:
        rows, fieldnames = _read_rows(path)
        loaded.append((path, rows, fieldnames))
        for row in rows:
            if not _is_jamendo(row):
                continue
            jamendo_rows += 1
            if _clean(row.get("release_date")):
                continue
            missing_before += 1
            jid = _jamendo_id(row)
            album_id = _album_id(row)
            if album_id:
                album_ids.add(album_id)
            elif jid:
                fallback_ids.add(jid)

    session = requests.Session()
    release_by_id: dict[str, dict[str, str]] = {}
    api_requests = 0
    album_list = sorted(album_ids, key=lambda value: int(value) if value.isdigit() else value)
    batch_size = max(1, int(album_batch_size))
    for start in range(0, len(album_list), batch_size):
        batch = album_list[start : start + batch_size]
        records, made = _fetch_album_batch(
            session,
            client_id=client_id,
            album_ids=batch,
            delay_seconds=delay_seconds,
        )
        release_by_id.update(records)
        api_requests += made
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        if (start // batch_size + 1) % 25 == 0:
            print(f"[INFO] album batches {start + len(batch)}/{len(album_list)}; resolved={len(release_by_id)}", flush=True)

    unresolved_ids: set[str] = set(fallback_ids)
    for _path, rows, _fieldnames in loaded:
        for row in rows:
            if not _is_jamendo(row) or _clean(row.get("release_date")):
                continue
            jid = _jamendo_id(row)
            if jid and jid not in release_by_id:
                unresolved_ids.add(jid)

    for pos, jid in enumerate(sorted(unresolved_ids, key=lambda value: int(value) if value.isdigit() else value), start=1):
        record, made = _fetch_track(session, client_id=client_id, jamendo_id=jid)
        api_requests += made
        if record.get("release_date"):
            release_by_id[jid] = record
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        if pos % 50 == 0:
            print(f"[INFO] fallback tracks {pos}/{len(unresolved_ids)}; resolved={len(release_by_id)}", flush=True)

    file_reports: list[dict[str, Any]] = []
    missing_after = 0
    total_updated = 0
    for path, rows, fieldnames in loaded:
        final_fields = _ensure_fields(fieldnames, ["album_id", "release_date", "release_year"])
        updated = 0
        file_missing_after = 0
        for row in rows:
            if not _is_jamendo(row):
                continue
            jid = _jamendo_id(row)
            record = release_by_id.get(jid, {})
            album_id = _album_id(row) or record.get("album_id", "")
            if album_id and not _clean(row.get("album_id")):
                row["album_id"] = album_id
                updated += 1
            if record.get("release_date") and not _clean(row.get("release_date")):
                row["release_date"] = record["release_date"]
                row["release_year"] = record.get("release_year") or _release_year(record["release_date"])
                updated += 1
            elif _clean(row.get("release_date")) and not _clean(row.get("release_year")):
                row["release_year"] = _release_year(row.get("release_date"))
                updated += 1
            if not _clean(row.get("release_date")):
                file_missing_after += 1
        if updated:
            _write_rows(path, rows, final_fields)
        total_updated += updated
        missing_after += file_missing_after
        file_reports.append({"path": str(path), "rows": len(rows), "updated_cells": updated, "missing_release_date": file_missing_after})

    return {
        "paths": [str(path) for path in paths],
        "jamendo_rows": jamendo_rows,
        "missing_release_date_before": missing_before,
        "missing_release_date_after": missing_after,
        "album_ids_queried": len(album_ids),
        "fallback_track_ids_queried": len(unresolved_ids),
        "release_records_resolved": len(release_by_id),
        "api_requests": api_requests,
        "updated_cells": total_updated,
        "files": file_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill Jamendo release_date/release_year fields from the Jamendo API.")
    parser.add_argument("--metadata", nargs="+", required=True, help="Metadata CSV path(s) to update in place.")
    parser.add_argument("--client_id", default="", help="Jamendo client id; defaults to JAMENDO_CLIENT_ID or run_jamendo_crawl.ps1.")
    parser.add_argument("--album_batch_size", type=int, default=10)
    parser.add_argument("--delay_seconds", type=float, default=0.12)
    args = parser.parse_args()

    client_id = _clean(args.client_id) or _load_client_id()
    if not client_id:
        raise SystemExit("Jamendo client id is required. Set JAMENDO_CLIENT_ID or pass --client_id.")
    report = enrich_files(
        [Path(path) for path in args.metadata],
        client_id=client_id,
        album_batch_size=args.album_batch_size,
        delay_seconds=args.delay_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
