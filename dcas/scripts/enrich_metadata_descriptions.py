from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"
JAMENDO_TRACKS_URL = "https://api.jamendo.com/v3.0/tracks"
WIKI_SEARCH_URL = "https://{lang}.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

ENRICHED_FIELDS = [
    "description",
    "album_description",
    "description_source",
    "album_description_source",
    "description_updated_at",
    "description_evidence_url",
    "tags",
    "musicinfo_language",
    "musicinfo_vocalinstrumental",
    "musicinfo_speed",
    "musicinfo_gender",
    "musicinfo_acoustic_electric",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fields})
    tmp.replace(path)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text


def _clip(text: str, limit: int = 900) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _first(*values: Any) -> str:
    for value in values:
        text = _clean(value)
        if text:
            return text
    return ""


def _has_description(row: dict[str, str]) -> bool:
    return bool(_first(row.get("description"), row.get("album_description")))


def _itunes_numeric_id(track_id: str) -> str:
    if track_id.startswith("itunes_"):
        return track_id.removeprefix("itunes_")
    if track_id.isdigit():
        return track_id
    return ""


def _flatten_jamendo_tags(musicinfo: dict[str, Any]) -> str:
    tags = musicinfo.get("tags") or {}
    found: list[str] = []
    if isinstance(tags, dict):
        for values in tags.values():
            if isinstance(values, list):
                found.extend(str(v) for v in values if _clean(v))
            elif _clean(values):
                found.append(str(values))
    return ",".join(dict.fromkeys(_clean(tag).lower() for tag in found if _clean(tag)))


def _load_jamendo_client_id() -> str:
    value = _clean(os.environ.get("JAMENDO_CLIENT_ID"))
    if value:
        return value
    run_script = Path("run_jamendo_crawl.ps1")
    if not run_script.exists():
        return ""
    text = run_script.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"\$JAMENDO_CLIENT_ID\s*=\s*\"([^\"]+)\"", text)
    if not match:
        return ""
    value = _clean(match.group(1))
    if not value or value.startswith("<") or value.endswith(">"):
        return ""
    return value


def _load_kimi_config() -> dict[str, str]:
    config: dict[str, str] = {}
    for path in (Path("configs/secrets/kimi.local.json"), Path("storage/secrets/kimi.local.json")):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        api_key = _clean(data.get("api_key") or data.get("apiKey"))
        if api_key:
            config["api_key"] = api_key
        model = _clean(data.get("model"))
        endpoint = _clean(data.get("endpoint"))
        if model:
            config["model"] = model
        if endpoint:
            config["endpoint"] = endpoint
        if config.get("api_key"):
            break
    if _clean(os.environ.get("KIMI_API_KEY")):
        config["api_key"] = _clean(os.environ.get("KIMI_API_KEY"))
    if _clean(os.environ.get("KIMI_MODEL")):
        config["model"] = _clean(os.environ.get("KIMI_MODEL"))
    if _clean(os.environ.get("KIMI_ENDPOINT")):
        config["endpoint"] = _clean(os.environ.get("KIMI_ENDPOINT"))
    config.setdefault("model", "kimi-k2.6")
    config.setdefault("endpoint", "https://api.moonshot.cn/v1/chat/completions")
    return config


def _select_candidates(
    rows: list[dict[str, str]],
    *,
    limit: int,
    per_culture: int,
    source_filter: set[str],
    overwrite_generated: bool,
    only_generated: bool,
) -> list[int]:
    missing = [
        i
        for i, row in enumerate(rows)
        if (
            (_clean(row.get("description_source")) == "kimi_generated")
            if only_generated
            else (
                not _has_description(row)
                or (overwrite_generated and _clean(row.get("description_source")) == "kimi_generated")
            )
        )
        and (not source_filter or _clean(row.get("source_dataset")).lower() in source_filter)
    ]
    if per_culture <= 0:
        return missing[:limit] if limit > 0 else missing

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx in missing:
        culture = _clean(rows[idx].get("culture")) or "unknown"
        if len(grouped[culture]) < per_culture:
            grouped[culture].append(idx)

    selected: list[int] = []
    cultures = sorted(grouped)
    for offset in range(per_culture):
        for culture in cultures:
            values = grouped[culture]
            if offset < len(values):
                selected.append(values[offset])
                if limit > 0 and len(selected) >= limit:
                    return selected
    return selected


def _lookup_itunes(row: dict[str, str], session: requests.Session) -> dict[str, Any]:
    tid = _itunes_numeric_id(_clean(row.get("track_id")))
    if not tid:
        return {}
    country = _clean(row.get("country")).upper() or "US"
    params = {"id": tid, "country": country, "entity": "song"}
    response = session.get(ITUNES_LOOKUP_URL, params=params, timeout=25)
    response.raise_for_status()
    data = response.json()
    for item in data.get("results") or []:
        if str(item.get("trackId")) == tid:
            return item
    return {}


def _apply_itunes_info(row: dict[str, str], item: dict[str, Any]) -> bool:
    changed = False
    genre = _clean(item.get("primaryGenreName"))
    if genre and not _clean(row.get("tags")):
        row["tags"] = genre
        changed = True
    release_date = _clean(item.get("releaseDate"))
    if release_date and not _clean(row.get("release_date")):
        row["release_date"] = release_date
        changed = True

    track_desc = _first(item.get("longDescription"), item.get("shortDescription"), item.get("description"))
    album_desc = _first(item.get("collectionDescription"))
    if track_desc and not _clean(row.get("description")):
        row["description"] = _clip(track_desc)
        row["description_source"] = "itunes_lookup"
        row["description_evidence_url"] = _first(item.get("trackViewUrl"), row.get("track_url"))
        changed = True
    if album_desc and not _clean(row.get("album_description")):
        row["album_description"] = _clip(album_desc)
        row["album_description_source"] = "itunes_lookup"
        row["description_evidence_url"] = _first(item.get("collectionViewUrl"), row.get("collection_url"))
        changed = True
    return changed


def _lookup_jamendo(row: dict[str, str], client_id: str, session: requests.Session) -> dict[str, Any]:
    tid = _clean(row.get("jamendo_id")) or _clean(row.get("track_id")).removeprefix("jamendo_")
    if not tid:
        return {}
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": 1,
        "id": tid,
        "include": "musicinfo+stats+lyrics",
    }
    response = session.get(JAMENDO_TRACKS_URL, params=params, timeout=25)
    response.raise_for_status()
    data = response.json()
    headers = data.get("headers") or {}
    if str(headers.get("status", "")).lower() == "failed":
        return {}
    results = data.get("results") or []
    return results[0] if results else {}


def _apply_jamendo_info(row: dict[str, str], item: dict[str, Any]) -> bool:
    changed = False
    musicinfo = item.get("musicinfo") if isinstance(item.get("musicinfo"), dict) else {}
    tags = _flatten_jamendo_tags(musicinfo)
    if tags and not _clean(row.get("tags")):
        row["tags"] = tags
        changed = True
    field_map = {
        "musicinfo_language": musicinfo.get("lang"),
        "musicinfo_vocalinstrumental": musicinfo.get("vocalinstrumental"),
        "musicinfo_speed": musicinfo.get("speed"),
        "musicinfo_gender": musicinfo.get("gender"),
        "musicinfo_acoustic_electric": musicinfo.get("acousticelectric"),
    }
    for key, value in field_map.items():
        text = _clean(value)
        if text and not _clean(row.get(key)):
            row[key] = text
            changed = True
    if _clean(item.get("releasedate")) and not _clean(row.get("release_date")):
        row["release_date"] = _clean(item.get("releasedate"))
        changed = True

    track_desc = _first(item.get("description"), item.get("shortdescription"))
    if track_desc and not _clean(row.get("description")):
        row["description"] = _clip(track_desc)
        row["description_source"] = "jamendo_api"
        row["description_evidence_url"] = _first(item.get("shareurl"), row.get("jamendo_url"))
        changed = True
    return changed


def _wiki_search(lang: str, query: str, session: requests.Session) -> dict[str, str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 2,
        "format": "json",
    }
    response = session.get(
        WIKI_SEARCH_URL.format(lang=lang),
        params=params,
        headers={"User-Agent": "EchoMusicResearch/0.1 metadata enrichment"},
        timeout=20,
    )
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    return hits[0] if hits else {}


def _wiki_summary(lang: str, title: str, session: requests.Session) -> dict[str, str]:
    url = WIKI_SUMMARY_URL.format(lang=lang, title=requests.utils.quote(title, safe=""))
    response = session.get(
        url,
        headers={"User-Agent": "EchoMusicResearch/0.1 metadata enrichment"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    extract = _clean(data.get("extract"))
    if not extract or len(extract) < 80:
        return {}
    if "may refer to" in extract.lower() or data.get("type") == "disambiguation":
        return {}
    return {
        "extract": extract,
        "url": _clean((data.get("content_urls") or {}).get("desktop", {}).get("page")),
        "title": _clean(data.get("title")),
    }


def _apply_wikipedia_description(
    row: dict[str, str],
    *,
    languages: list[str],
    session: requests.Session,
) -> bool:
    title = _clean(row.get("title"))
    artist = _clean(row.get("artist"))
    album = _clean(row.get("album"))
    queries: list[tuple[str, str]] = []
    if album and artist:
        queries.append(("album_description", f"{album} {artist} album"))
    if title and artist:
        queries.append(("description", f"{title} {artist} song"))

    for field, query in queries:
        if _clean(row.get(field)):
            continue
        for lang in languages:
            try:
                hit = _wiki_search(lang, query, session)
                if not hit:
                    continue
                summary = _wiki_summary(lang, _clean(hit.get("title")), session)
            except requests.RequestException:
                continue
            if not summary:
                continue
            row[field] = _clip(summary["extract"], 700)
            if field == "album_description":
                row["album_description_source"] = f"wikipedia:{lang}"
            else:
                row["description_source"] = f"wikipedia:{lang}"
            row["description_evidence_url"] = summary.get("url", "")
            return True
    return False


def _kimi_prompt(row: dict[str, str]) -> str:
    fields = {
        "title": _clean(row.get("title")),
        "artist": _clean(row.get("artist")),
        "album": _clean(row.get("album")),
        "culture": _clean(row.get("culture")),
        "country": _clean(row.get("country")),
        "release_date": _clean(row.get("release_date")),
        "tags": _clean(row.get("tags")),
        "musicinfo_language": _clean(row.get("musicinfo_language")),
        "musicinfo_vocalinstrumental": _clean(row.get("musicinfo_vocalinstrumental")),
        "musicinfo_speed": _clean(row.get("musicinfo_speed")),
        "source": _clean(row.get("source_dataset")),
    }
    compact = json.dumps(fields, ensure_ascii=False)
    return (
        "Write one concise Chinese paragraph for a music recommendation card. "
        "Length 90-150 Chinese characters. Be concrete about likely sound, mood, "
        "era, genre, or instrumentation, but do not invent awards, chart positions, "
        "personal biographies, exact release-market claims, or artist nationality. "
        "The country and culture fields are catalog tags, not proof of artist origin "
        "or release territory. If evidence is thin, use cautious wording such as "
        "\"classified under\" or \"the metadata points to\". Return only the paragraph, "
        "no markdown.\n\n"
        f"Metadata: {compact}"
    )


def _call_kimi(row: dict[str, str], config: dict[str, str]) -> str:
    api_key = _clean(config.get("api_key"))
    if not api_key:
        return ""
    endpoint = _clean(config.get("endpoint")) or "https://api.moonshot.cn/v1/chat/completions"
    for token_budget in (1024, 1536):
        payload = {
            "model": _clean(config.get("model")) or "kimi-k2.6",
            "messages": [
                {
                    "role": "system",
                    "content": "You write careful, non-hallucinated Chinese music metadata blurbs.",
                },
                {"role": "user", "content": _kimi_prompt(row)},
            ],
            "max_completion_tokens": token_budget,
            "thinking": {"type": "disabled"},
        }
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = _clean(data.get("choices", [{}])[0].get("message", {}).get("content"))
        if text:
            return _clip(text, 450)
    return ""


def enrich_metadata_descriptions(
    metadata_csv: str | Path,
    out_csv: str | Path | None = None,
    *,
    limit: int = 0,
    per_culture: int = 0,
    use_itunes: bool = True,
    use_jamendo: bool = True,
    use_wikipedia: bool = True,
    use_kimi: bool = True,
    max_wikipedia: int = 80,
    max_kimi: int = 40,
    kimi_workers: int = 3,
    sleep_seconds: float = 0.15,
    source_filter: set[str] | None = None,
    overwrite_generated: bool = False,
    only_generated: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    in_path = Path(metadata_csv)
    out_path = Path(out_csv) if out_csv else in_path
    rows, fields = _read_rows(in_path)
    for field in ENRICHED_FIELDS:
        if field not in fields:
            fields.append(field)

    candidates = _select_candidates(
        rows,
        limit=limit,
        per_culture=per_culture,
        source_filter=source_filter or set(),
        overwrite_generated=overwrite_generated,
        only_generated=only_generated,
    )
    session = requests.Session()
    jamendo_client_id = _load_jamendo_client_id()
    kimi_config = _load_kimi_config()
    wiki_languages = ["en", "zh", "ja", "es", "pt", "de", "ko"]
    changed_indices: set[int] = set()
    stats: dict[str, Any] = {
        "metadata": str(in_path.resolve()),
        "out_csv": str(out_path.resolve()),
        "rows": len(rows),
        "candidate_rows": len(candidates),
        "itunes_checked": 0,
        "jamendo_checked": 0,
        "platform_descriptions": 0,
        "platform_metadata_updates": 0,
        "wikipedia_descriptions": 0,
        "kimi_descriptions": 0,
        "kimi_available": bool(kimi_config.get("api_key")),
        "jamendo_available": bool(jamendo_client_id),
        "source_filter": sorted(source_filter or []),
        "overwrite_generated": overwrite_generated,
        "only_generated": only_generated,
    }

    for pos, idx in enumerate(candidates, start=1):
        row = rows[idx]
        before = dict(row)
        source = _clean(row.get("source_dataset")).lower()
        try:
            if use_itunes and source == "itunes":
                item = _lookup_itunes(row, session)
                stats["itunes_checked"] += 1
                if item and _apply_itunes_info(row, item):
                    changed_indices.add(idx)
            elif use_jamendo and source == "jamendo" and jamendo_client_id:
                item = _lookup_jamendo(row, jamendo_client_id, session)
                stats["jamendo_checked"] += 1
                if item and _apply_jamendo_info(row, item):
                    changed_indices.add(idx)
        except requests.RequestException as exc:
            print(f"[WARN] platform lookup failed for row {idx}: {exc}", flush=True)
        if row != before:
            stats["platform_metadata_updates"] += 1
            if _clean(row.get("description")) or _clean(row.get("album_description")):
                stats["platform_descriptions"] += 1
        if pos % 25 == 0:
            print(f"[INFO] platform pass {pos}/{len(candidates)}", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    wiki_count = 0
    if use_wikipedia and max_wikipedia > 0:
        for idx in candidates:
            if wiki_count >= max_wikipedia:
                break
            row = rows[idx]
            if _has_description(row):
                continue
            if _apply_wikipedia_description(row, languages=wiki_languages, session=session):
                row["description_updated_at"] = _utc_now()
                changed_indices.add(idx)
                wiki_count += 1
                stats["wikipedia_descriptions"] = wiki_count
                if wiki_count % 10 == 0:
                    print(f"[INFO] wikipedia descriptions {wiki_count}/{max_wikipedia}", flush=True)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    kimi_targets = [
        idx
        for idx in candidates
        if not _has_description(rows[idx])
        or (overwrite_generated and _clean(rows[idx].get("description_source")) == "kimi_generated")
    ][: max(0, max_kimi)]
    if use_kimi and kimi_targets and kimi_config.get("api_key"):
        def run_one(row_idx: int) -> tuple[int, str, str]:
            for attempt in range(1, 3):
                try:
                    return row_idx, _call_kimi(rows[row_idx], kimi_config), ""
                except requests.RequestException as exc:
                    if attempt >= 2:
                        return row_idx, "", str(exc)
                    time.sleep(1.5 * attempt)
            return row_idx, "", "unknown"

        with ThreadPoolExecutor(max_workers=max(1, int(kimi_workers))) as pool:
            futures = [pool.submit(run_one, idx) for idx in kimi_targets]
            for done, future in enumerate(as_completed(futures), start=1):
                idx, text, error = future.result()
                if error:
                    print(f"[WARN] kimi failed for row {idx}: {error[:180]}", flush=True)
                if text and (
                    not _clean(rows[idx].get("description"))
                    or (overwrite_generated and _clean(rows[idx].get("description_source")) == "kimi_generated")
                ):
                    rows[idx]["description"] = text
                    rows[idx]["description_source"] = "kimi_generated"
                    rows[idx]["description_updated_at"] = _utc_now()
                    changed_indices.add(idx)
                    stats["kimi_descriptions"] += 1
                if done % 10 == 0:
                    print(f"[INFO] kimi descriptions {done}/{len(kimi_targets)}", flush=True)

    now = _utc_now()
    for idx in changed_indices:
        if not _clean(rows[idx].get("description_updated_at")):
            rows[idx]["description_updated_at"] = now

    stats["changed_rows"] = len(changed_indices)
    stats["rows_with_description"] = sum(1 for row in rows if _clean(row.get("description")))
    stats["rows_with_album_description"] = sum(1 for row in rows if _clean(row.get("album_description")))
    stats["rows_with_tags"] = sum(1 for row in rows if _clean(row.get("tags")))

    if not dry_run:
        _write_rows(out_path, rows, fields)
        report_path = out_path.with_suffix(out_path.suffix + ".description_report.json")
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["report_path"] = str(report_path.resolve())
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich merged music metadata with descriptions and source evidence.")
    ap.add_argument("--metadata", default="storage/public/merged/metadata_merged.csv")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0, help="Max missing-description rows to process; 0 means all.")
    ap.add_argument("--per_culture", type=int, default=0, help="Round-robin at most N rows per culture before applying --limit.")
    ap.add_argument("--no_itunes", action="store_true")
    ap.add_argument("--no_jamendo", action="store_true")
    ap.add_argument("--no_wikipedia", action="store_true")
    ap.add_argument("--no_kimi", action="store_true")
    ap.add_argument("--max_wikipedia", type=int, default=80)
    ap.add_argument("--max_kimi", type=int, default=40)
    ap.add_argument("--kimi_workers", type=int, default=3)
    ap.add_argument("--sleep_seconds", type=float, default=0.15)
    ap.add_argument("--source", default="", help="Optional comma-separated source_dataset filter, e.g. itunes,jamendo.")
    ap.add_argument("--overwrite_generated", action="store_true", help="Regenerate rows whose description_source is kimi_generated.")
    ap.add_argument("--only_generated", action="store_true", help="Only process rows whose description_source is kimi_generated.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    source_filter = {s.strip().lower() for s in args.source.split(",") if s.strip()}

    report = enrich_metadata_descriptions(
        metadata_csv=args.metadata,
        out_csv=args.out or None,
        limit=args.limit,
        per_culture=args.per_culture,
        use_itunes=not args.no_itunes,
        use_jamendo=not args.no_jamendo,
        use_wikipedia=not args.no_wikipedia,
        use_kimi=not args.no_kimi,
        max_wikipedia=args.max_wikipedia,
        max_kimi=args.max_kimi,
        kimi_workers=args.kimi_workers,
        sleep_seconds=args.sleep_seconds,
        source_filter=source_filter,
        overwrite_generated=args.overwrite_generated,
        only_generated=args.only_generated,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
