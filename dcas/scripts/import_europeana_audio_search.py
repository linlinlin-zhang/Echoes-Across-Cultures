from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


def _slug(v: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(v).strip()).strip("_") or "item"


def _is_allowed_rights(rights_url: str) -> bool:
    s = str(rights_url).lower().strip()
    allowed_tokens = [
        "creativecommons.org/licenses/by/",
        "creativecommons.org/licenses/by-sa/",
        "creativecommons.org/licenses/by-nc/",
        "creativecommons.org/licenses/by-nc-sa/",
        "creativecommons.org/publicdomain/",
        "rightsstatements.org/vocab/pdm",
        "rightsstatements.org/vocab/no",
        "rightsstatements.org/vocab/inc",
    ]
    return any(tok in s for tok in allowed_tokens)


def _flatten_first(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _flatten_join(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    return text


def _extract_candidate_audio_urls(item: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("edmIsShownBy", "edmHasView"):
        value = item.get(key, [])
        if isinstance(value, list):
            for candidate in value:
                text = str(candidate).strip()
                if text and text not in urls:
                    urls.append(text)
    return urls


def _guess_extension(url: str, content_type: str, first_bytes: bytes) -> str:
    lower_ct = str(content_type).lower()
    lower_url = str(url).lower()
    if "mpeg" in lower_ct or lower_url.endswith(".mp3") or first_bytes.startswith(b"ID3"):
        return ".mp3"
    if "wav" in lower_ct or lower_url.endswith(".wav") or first_bytes.startswith(b"RIFF"):
        return ".wav"
    if "ogg" in lower_ct or lower_url.endswith(".ogg") or first_bytes.startswith(b"OggS"):
        return ".ogg"
    if "flac" in lower_ct or lower_url.endswith(".flac") or first_bytes.startswith(b"fLaC"):
        return ".flac"
    if "mp4" in lower_ct or "m4a" in lower_ct or lower_url.endswith(".m4a"):
        return ".m4a"
    return ".bin"


def _looks_like_audio(content_type: str, first_bytes: bytes) -> bool:
    lower_ct = str(content_type).lower()
    if lower_ct.startswith("audio/"):
        return True
    return any(first_bytes.startswith(prefix) for prefix in (b"ID3", b"RIFF", b"OggS", b"fLaC"))


def _download_audio_with_redundancy(
    session: requests.Session,
    candidate_urls: list[str],
    out_base: Path,
    request_timeout: int = 60,
    max_attempts_per_url: int = 2,
) -> tuple[Path, str, str]:
    errors: list[str] = []
    for url in candidate_urls:
        for attempt in range(1, int(max_attempts_per_url) + 1):
            try:
                with session.get(url, timeout=request_timeout, stream=True, allow_redirects=True) as resp:
                    resp.raise_for_status()
                    first_chunk = b""
                    chunks: list[bytes] = []
                    for chunk in resp.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        if not first_chunk:
                            first_chunk = bytes(chunk[:16])
                        chunks.append(bytes(chunk))
                    content_type = str(resp.headers.get("Content-Type", "")).strip()
                    if not chunks:
                        raise RuntimeError("empty response body")
                    if not _looks_like_audio(content_type=content_type, first_bytes=first_chunk):
                        raise RuntimeError(f"non-audio response content_type={content_type!r}")
                    ext = _guess_extension(url=resp.url, content_type=content_type, first_bytes=first_chunk)
                    out_path = out_base.with_suffix(ext)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "wb") as f:
                        for chunk in chunks:
                            f.write(chunk)
                    return out_path, str(resp.url), content_type
            except Exception as e:
                errors.append(f"url={url} attempt={attempt}: {e}")
                time.sleep(0.5 * attempt)
    raise RuntimeError("; ".join(errors[:8]) or "all candidate audio downloads failed")


def import_europeana_audio_search(
    query: str,
    out_dir: str | Path,
    culture: str,
    limit: int = 50,
    wskey: str = "api2demo",
    rows_per_page: int = 100,
    use_cursor: bool = True,
    query_filters: list[str] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = out / "metadata.csv"
    report_json = out / "import_report.json"

    session = requests.Session()
    session.headers.update({"User-Agent": "Echo/DCAS Europeana Importer"})

    imported = 0
    scanned = 0
    page = 1
    cursor: str | None = "*" if use_cursor else None
    rows: list[dict[str, str]] = []
    errors: list[str] = []

    while imported < int(limit):
        params = {
            "wskey": wskey,
            "query": query,
            "rows": int(rows_per_page),
            "profile": "standard",
        }
        if query_filters:
            params["qf"] = [str(x) for x in query_filters if str(x).strip()]
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params["start"] = int((page - 1) * rows_per_page) + 1
        r = session.get("https://api.europeana.eu/record/v2/search.json", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            if imported >= int(limit):
                break
            scanned += 1
            try:
                if str(item.get("type", "")).upper() != "SOUND":
                    continue
                rights = [str(x) for x in item.get("rights", [])]
                if not rights or not all(_is_allowed_rights(x) for x in rights):
                    continue
                candidate_urls = _extract_candidate_audio_urls(item)
                if not candidate_urls:
                    continue

                item_id = str(item.get("id", "")).strip("/") or f"row_{scanned:06d}"
                track_id = _slug(item_id.replace("/", "_"))
                audio_path, final_audio_url, content_type = _download_audio_with_redundancy(
                    session=session,
                    candidate_urls=candidate_urls,
                    out_base=audio_dir / track_id,
                )

                title = ""
                if isinstance(item.get("title"), list) and item.get("title"):
                    title = str(item["title"][0])

                country = _flatten_first(item.get("country", []))
                data_provider = _flatten_first(item.get("dataProvider", []))
                provider = _flatten_first(item.get("provider", []))
                language = _flatten_join(item.get("language", []))
                collection = _flatten_join(item.get("europeanaCollectionName", []))
                dataset_name = _flatten_join(item.get("edmDatasetName", []))
                shown_at = _flatten_join(item.get("edmIsShownAt", []))

                note_parts = [
                    f"country={country}" if country else "",
                    f"data_provider={data_provider}" if data_provider else "",
                    f"provider={provider}" if provider else "",
                    f"collection={collection}" if collection else "",
                    f"dataset={dataset_name}" if dataset_name else "",
                    f"content_type={content_type}" if content_type else "",
                ]

                rows.append(
                    {
                        "track_id": track_id,
                        "culture": str(culture),
                        "audio_path": str(Path("audio") / audio_path.name),
                        "source_dataset": "europeana_search",
                        "source_split": "search",
                        "source_index": str(scanned - 1),
                        "search_query": query,
                        "query_filters": ";".join(str(x) for x in (query_filters or []) if str(x).strip()),
                        "label": "",
                        "region": country,
                        "language": language,
                        "title": title,
                        "license": ";".join(rights),
                        "license_note": "Imported from Europeana Search API with rights filtering and redundant audio resolution.",
                        "url": str(item.get("guid", "")),
                        "notes": " | ".join(part for part in note_parts if part),
                        "country": country,
                        "data_provider": data_provider,
                        "provider": provider,
                        "europeana_collection": collection,
                        "europeana_dataset_name": dataset_name,
                        "edm_is_shown_by": final_audio_url,
                        "edm_is_shown_at": shown_at,
                    }
                )
                imported += 1
            except Exception as e:
                errors.append(f"row={scanned}: {e}")

        if cursor is not None:
            cursor = data.get("nextCursor")
            if not cursor:
                break
        else:
            page += 1

    with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "culture",
                "audio_path",
                "source_dataset",
                "source_split",
                "source_index",
                "search_query",
                "query_filters",
                "label",
                "region",
                "language",
                "title",
                "license",
                "license_note",
                "url",
                "notes",
                "country",
                "data_provider",
                "provider",
                "europeana_collection",
                "europeana_dataset_name",
                "edm_is_shown_by",
                "edm_is_shown_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "query": query,
        "culture": culture,
        "wskey": wskey,
        "requested_limit": int(limit),
        "rows_per_page": int(rows_per_page),
        "use_cursor": bool(use_cursor),
        "query_filters": [str(x) for x in (query_filters or []) if str(x).strip()],
        "scanned": int(scanned),
        "imported": int(imported),
        "errors": errors[:200],
        "metadata_csv": str(metadata_csv.resolve()),
        "audio_dir": str(audio_dir.resolve()),
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Import audio records from a Europeana search query.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--culture", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--wskey", default="api2demo")
    ap.add_argument("--rows_per_page", type=int, default=100)
    ap.add_argument("--no_cursor", action="store_true")
    ap.add_argument(
        "--query_filter",
        action="append",
        default=[],
        help="Optional repeated Europeana qf filter, e.g. TYPE:SOUND or COUNTRY:France",
    )
    args = ap.parse_args()

    out = import_europeana_audio_search(
        query=str(args.query),
        out_dir=str(args.out_dir),
        culture=str(args.culture),
        limit=int(args.limit),
        wskey=str(args.wskey),
        rows_per_page=int(args.rows_per_page),
        use_cursor=not bool(args.no_cursor),
        query_filters=[str(x) for x in args.query_filter if str(x).strip()],
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
