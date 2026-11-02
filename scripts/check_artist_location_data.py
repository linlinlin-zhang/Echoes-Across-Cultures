"""Check the published location dictionary and its catalog coverage without network calls."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import re

from enrich_artist_locations import ROOT, artist_key, atomic


def main():
    path = ROOT / "storage/public/merged/artist_locations.json"
    data = json.loads(path.read_text())
    report = data["report"]
    capitals = json.loads((ROOT / "storage/artist_location_cache/capitals.json").read_text())
    rows = list(csv.DictReader((path.parent / "metadata_merged.csv").open(encoding="utf-8-sig", newline="")))
    assert len(rows) == len({r["track_id"] for r in rows}) == 30000
    assert report == json.loads(path.with_suffix(".report.json").read_text())
    for key, location in data["artists"].items():
        assert re.fullmatch(r"[A-Z]{2}", location["country"]), key
        assert location.get("city", "").strip(), (key, "missing city label")
        assert location["location_precision"] in {"artist_city", "country_capital"}, key
        assert location["location_source_url"].startswith("https://"), key
        if location.get("location_place_url"):
            assert location["location_place_url"].startswith("https://"), key
        else:
            assert location.get("location_relation") == "public_artist_profile" and location.get("declared_city"), key
        for coord, bound in [("lat", 90), ("lng", 180)]:
            number = location[coord]
            assert isinstance(number, (int, float)) and not isinstance(number, bool), (key, coord)
            assert math.isfinite(number) and -bound <= number <= bound, (key, coord)
        if location["location_precision"] == "country_capital":
            expected = capitals[location["country"]]["point"]
            assert all(abs(location[c] - v) < 1e-7 for c, v in zip(["lat", "lng"], expected)), key
    groups = defaultdict(list)
    missing = Counter()
    precision = Counter()
    for row in rows:
        key = artist_key(row)
        groups[key].append(row)
        location = data["artists"].get(data["tracks"].get(row["track_id"])) or data["artists"].get(key)
        if location:
            precision[location["location_precision"]] += 1
        else:
            missing[key] += 1
    assert len(groups) == report["artist_identities"]
    assert set(missing) == set(data["unresolved"])
    assert len(missing) == report["unresolved_catalog_artist_groups"]
    assert len(groups) - len(missing) == report["fully_located_catalog_artist_groups"]
    assert dict(precision) == report["track_precision_counts"]
    assert sum(precision.values()) == report["tracks_with_artist_locations"]
    assert sum(missing.values()) == report["tracks_without_verified_artist_location"]
    queue = list(csv.DictReader(path.with_suffix(".unresolved.csv").open(encoding="utf-8-sig", newline="")))
    assert {r["artist_key"]: int(r["remaining_tracks"]) for r in queue} == dict(missing)
    assert [int(r["remaining_tracks"]) for r in queue] == sorted(missing.values(), reverse=True)
    translations = path.parent / "descriptions_en.json"
    translation_hash = hashlib.sha256(translations.read_bytes()).hexdigest()
    assert translation_hash == "1280245817e689e5537da28ac5910aa816f4ff02c1aa4ca2754aefdd6a865eaf"
    result = {
        "catalog_tracks": len(rows),
        **report,
        "description_dictionary_sha256": translation_hash,
        "checks": [
            "finite bounded coordinates",
            "public location sources",
            "capital fallback consistency",
            "all catalog mappings and coverage totals",
            "complete sorted unresolved queue",
            "unchanged bilingual description dictionary",
        ],
    }
    atomic(ROOT / "tmp/web-self-check/location-data-validation.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
