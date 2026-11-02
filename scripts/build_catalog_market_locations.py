"""Use the platform's own market only for tracks without verified artist geography."""

from __future__ import annotations
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from enrich_artist_locations import ROOT, artist_key, atomic


def main():
    folder = ROOT / "storage/public/merged"
    data = json.loads((folder / "artist_locations.json").read_text(encoding="utf-8"))
    capitals = json.loads((ROOT / "storage/artist_location_cache/capitals.json").read_text(encoding="utf-8"))
    reference_text = (ROOT / "web/geo-data.js").read_text(encoding="utf-8")
    regions = json.loads(reference_text.split("window.EchoGeoData = ", 1)[1].strip().rstrip(";"))
    rows = list(csv.DictReader((folder / "metadata_merged.csv").open(encoding="utf-8-sig", newline="")))
    result, unavailable = {}, []
    for row in rows:
        tid = row["track_id"]
        if data["artists"].get(data["tracks"].get(tid)) or data["artists"].get(artist_key(row)):
            continue
        code = row.get("country", "").strip().upper()
        capital = capitals.get(code)
        if not capital and code in regions:
            region = regions[code]
            capital = {
                "labels": {"en": region["name"]},
                "point": list(reversed(region["point"])),
                "url": "https://www.wikidata.org/wiki/Q8646" if code == "HK" else "https://www.naturalearthdata.com/",
            }
        if not capital:
            unavailable.append(
                {"track_id": tid, "artist": row.get("artist", ""), "reason": "platform_country_unavailable"}
            )
            continue
        source = next(
            (
                row.get(k, "").strip()
                for k in ("platform_track_url", "track_url", "jamendo_url", "artist_url")
                if row.get(k, "").strip().startswith("https://")
            ),
            "",
        )
        if not source:
            unavailable.append(
                {"track_id": tid, "artist": row.get("artist", ""), "reason": "platform_source_unavailable"}
            )
            continue
        result[tid] = {
            "country": code,
            "city": capital["labels"].get("en", ""),
            "city_labels": capital["labels"],
            "lat": capital["point"][0],
            "lng": capital["point"][1],
            "location_precision": "catalog_market",
            "country_source": "catalog_market",
            "location_relation": "platform_catalog_market",
            "location_source_url": source,
            "location_place_url": capital["url"].replace("http:", "https:").replace("/entity/", "/wiki/"),
        }
    report = {
        "catalog_tracks": len(rows),
        "verified_artist_tracks": data["report"]["tracks_with_artist_locations"],
        "platform_market_tracks": len(result),
        "tracks_with_display_location": data["report"]["tracks_with_artist_locations"] + len(result),
        "platform_without_location": len(unavailable),
        "fallback_countries": dict(sorted(Counter(v["country"] for v in result.values()).items())),
    }
    atomic(
        folder / "catalog_market_locations.json",
        {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "method": "Original platform catalog country; capital or region center is a display reference, not artist location",
            "report": report,
            "tracks": result,
            "unavailable": unavailable,
        },
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
