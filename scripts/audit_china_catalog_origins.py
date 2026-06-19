from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas_server.catalog_geo import infer_catalog_origin


CHINA_FAMILY_ISOS = {"CN", "HK", "MO", "TW"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def is_china_marked(row: dict[str, Any]) -> bool:
    country = clean(row.get("country")).upper()
    culture = clean(row.get("culture")).casefold()
    return country in CHINA_FAMILY_ISOS or culture == "china"


def review_status(origin: dict[str, Any]) -> str:
    iso = clean(origin.get("country_iso")).upper()
    source = clean(origin.get("country_source"))
    if iso in CHINA_FAMILY_ISOS:
        return "confirmed_china_family"
    if iso:
        return "corrected_to_other_country"
    if source == "itunes_storefront_china_review":
        return "rejected_china_storefront_needs_review"
    return "unknown"


def review_reason(row: dict[str, Any], origin: dict[str, Any]) -> str:
    source = clean(origin.get("country_source"))
    iso = clean(origin.get("country_iso")).upper()
    if source == "artist_origin_hint":
        return f"artist override or known artist origin -> {iso}"
    if source == "genre_or_metadata_hint":
        return f"genre/label/tags/description evidence -> {iso}"
    if source == "itunes_storefront":
        return f"non-China iTunes storefront retained -> {iso}"
    if source == "itunes_storefront_china_review":
        return "China-family iTunes storefront only; no strong artist or metadata evidence"
    return source or "no usable origin evidence"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit catalog rows marked as China-family origin.")
    parser.add_argument(
        "--metadata",
        default="storage/public/merged/metadata_merged_local_audio.csv",
        help="Merged metadata CSV to audit.",
    )
    parser.add_argument(
        "--output",
        default="reports/catalog_geo/china_origin_audit.csv",
        help="Destination CSV report.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not is_china_marked(row):
                continue
            origin = infer_catalog_origin(row)
            rows.append(
                {
                    "track_id": clean(row.get("track_id")),
                    "title": clean(row.get("title")),
                    "artist": clean(row.get("artist")),
                    "album": clean(row.get("album")),
                    "raw_country": clean(row.get("country")),
                    "raw_culture": clean(row.get("culture")),
                    "source_dataset": clean(row.get("source_dataset")),
                    "label": clean(row.get("label")),
                    "tags": clean(row.get("tags")),
                    "description": clean(row.get("description")),
                    "inferred_country": clean(origin.get("country")),
                    "inferred_country_iso": clean(origin.get("country_iso")),
                    "country_source": clean(origin.get("country_source")),
                    "storefront_country": clean(origin.get("storefront_country")),
                    "review_status": review_status(origin),
                    "review_reason": review_reason(row, origin),
                    "platform_track_url": clean(row.get("platform_track_url") or row.get("itunes_url")),
                }
            )

    fieldnames = [
        "track_id",
        "title",
        "artist",
        "album",
        "raw_country",
        "raw_culture",
        "source_dataset",
        "label",
        "tags",
        "description",
        "inferred_country",
        "inferred_country_iso",
        "country_source",
        "storefront_country",
        "review_status",
        "review_reason",
        "platform_track_url",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["review_status"] for row in rows)
    iso_counts = Counter(row["inferred_country_iso"] or "(blank)" for row in rows)
    print(f"audit_rows={len(rows)}")
    print("review_status=" + ", ".join(f"{key}:{value}" for key, value in status_counts.most_common()))
    print("inferred_iso=" + ", ".join(f"{key}:{value}" for key, value in iso_counts.most_common()))
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
