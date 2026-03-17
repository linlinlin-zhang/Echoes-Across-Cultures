from __future__ import annotations

import argparse
import csv
import json
import re
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


def import_europeana_audio_search(
    query: str,
    out_dir: str | Path,
    culture: str,
    limit: int = 50,
    wskey: str = "api2demo",
    rows_per_page: int = 100,
    use_cursor: bool = True,
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
                shown_by = [str(x) for x in item.get("edmIsShownBy", [])]
                if not shown_by:
                    continue
                audio_url = shown_by[0]
                if not audio_url.lower().endswith(".mp3"):
                    continue

                item_id = str(item.get("id", "")).strip("/") or f"row_{scanned:06d}"
                track_id = _slug(item_id.replace("/", "_"))
                audio_path = audio_dir / f"{track_id}.mp3"

                ar = session.get(audio_url, timeout=60)
                ar.raise_for_status()
                with open(audio_path, "wb") as f:
                    f.write(ar.content)

                title = ""
                if isinstance(item.get("title"), list) and item.get("title"):
                    title = str(item["title"][0])

                rows.append(
                    {
                        "track_id": track_id,
                        "culture": str(culture),
                        "audio_path": str(Path("audio") / audio_path.name),
                        "source_dataset": "europeana_search",
                        "source_split": "search",
                        "source_index": str(scanned - 1),
                        "label": "",
                        "title": title,
                        "rights": ";".join(rights),
                        "source_url": str(item.get("guid", "")),
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
                "label",
                "title",
                "rights",
                "source_url",
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
    args = ap.parse_args()

    out = import_europeana_audio_search(
        query=str(args.query),
        out_dir=str(args.out_dir),
        culture=str(args.culture),
        limit=int(args.limit),
        wskey=str(args.wskey),
        rows_per_page=int(args.rows_per_page),
        use_cursor=not bool(args.no_cursor),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
