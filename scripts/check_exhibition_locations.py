"""Check artist-first and platform-market fallback against every catalog row."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcas_server.artist_locations import apply_artist_location
from enrich_artist_locations import artist_key, atomic


def main():
    folder = ROOT / "storage/public/merged"
    artists = json.loads((folder / "artist_locations.json").read_text(encoding="utf-8"))
    fallback = json.loads((folder / "catalog_market_locations.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((folder / "metadata_merged.csv").open(encoding="utf-8-sig", newline="")))
    counts = {"artist_location": 0, "catalog_market": 0, "unknown": 0}
    for row in rows:
        record = artists["artists"].get(artists["tracks"].get(row["track_id"])) or artists["artists"].get(
            artist_key(row)
        )
        item = apply_artist_location(dict(row), ROOT / "storage")
        if record:
            assert all(item[k] == record[k] for k in ["country", "city", "lat", "lng", "location_precision"])
            counts["artist_location"] += 1
        elif row["track_id"] in fallback["tracks"]:
            assert item["country"] == row["country"] == item["catalog_country"]
            assert item["location_precision"] == item["country_source"] == "catalog_market"
            assert item["location_source_url"].startswith("https://")
            assert -90 <= item["lat"] <= 90 and -180 <= item["lng"] <= 180
            counts["catalog_market"] += 1
        else:
            assert not row["country"]
            counts["unknown"] += 1
    assert counts == {"artist_location": 25996, "catalog_market": 3961, "unknown": 43}, counts
    atomic(ROOT / "tmp/exhibition-20261102/location-validation.json", counts)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
