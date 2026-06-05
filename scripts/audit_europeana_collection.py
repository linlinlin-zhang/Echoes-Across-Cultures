from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import requests


def _is_open_rights(rights: str) -> bool:
    s = str(rights).lower()
    return any(
        token in s
        for token in [
            "creativecommons.org/licenses/by/",
            "creativecommons.org/licenses/by-sa/",
            "creativecommons.org/publicdomain/",
            "rightsstatements.org/vocab/pdm",
            "rightsstatements.org/vocab/no",
            "rightsstatements.org/vocab/inc",
        ]
    )


def audit_collection(
    query: str,
    out_json: str | Path,
    out_csv: str | Path | None = None,
    wskey: str = "api2demo",
    rows: int = 50,
) -> dict[str, Any]:
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
        "wskey": wskey,
        "query": query,
        "rows": int(rows),
        "profile": "standard",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    items: list[dict[str, Any]] = []
    for item in data.get("items", []):
        rights_list = [str(x) for x in item.get("rights", [])]
        record = {
            "id": str(item.get("id", "")),
            "guid": str(item.get("guid", "")),
            "title": "; ".join(str(x) for x in item.get("title", [])) if isinstance(item.get("title"), list) else "",
            "country": "; ".join(str(x) for x in item.get("country", [])),
            "data_provider": "; ".join(str(x) for x in item.get("dataProvider", [])),
            "type": str(item.get("type", "")),
            "shown_at": "; ".join(str(x) for x in item.get("edmIsShownAt", [])),
            "shown_by": "; ".join(str(x) for x in item.get("edmIsShownBy", [])),
            "rights": "; ".join(rights_list),
            "preview_no_distribute": bool(item.get("previewNoDistribute", False)),
            "has_audio_proxy": bool(item.get("edmIsShownBy")),
            "has_landing_page": bool(item.get("edmIsShownAt")),
            "open_rights_heuristic": all(_is_open_rights(x) for x in rights_list) if rights_list else False,
        }
        items.append(record)

    summary = {
        "query": query,
        "requested_rows": int(rows),
        "returned_rows": int(len(items)),
        "items_count_reported": int(data.get("itemsCount", len(items))),
        "total_results": int(data.get("totalResults", 0)),
        "n_has_audio_proxy": int(sum(1 for x in items if x["has_audio_proxy"])),
        "n_has_landing_page": int(sum(1 for x in items if x["has_landing_page"])),
        "n_open_rights_heuristic": int(sum(1 for x in items if x["open_rights_heuristic"])),
        "n_sound_type": int(sum(1 for x in items if str(x["type"]).upper() == "SOUND")),
    }

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": items}, f, ensure_ascii=False, indent=2)

    if out_csv is not None:
        out_csv_path = Path(out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "guid",
                    "title",
                    "country",
                    "data_provider",
                    "type",
                    "shown_at",
                    "shown_by",
                    "rights",
                    "preview_no_distribute",
                    "has_audio_proxy",
                    "has_landing_page",
                    "open_rights_heuristic",
                ],
            )
            writer.writeheader()
            writer.writerows(items)

    return {
        "summary": summary,
        "out_json": str(out_json_path),
        "out_csv": str(out_csv) if out_csv else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit Europeana collection search results for rights and audio proxy fields."
    )
    ap.add_argument("--query", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--wskey", default="api2demo")
    ap.add_argument("--rows", type=int, default=50)
    args = ap.parse_args()

    out = audit_collection(
        query=str(args.query),
        out_json=str(args.out_json),
        out_csv=str(args.out_csv) if args.out_csv else None,
        wskey=str(args.wskey),
        rows=int(args.rows),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
