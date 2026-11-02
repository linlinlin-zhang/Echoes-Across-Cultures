"""Resolve self-declared Jamendo cities using country-constrained Wikidata place names."""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from enrich_artist_locations import ROOT, fetch_all, batches, atomic, point, value


def clean(value):
    return str(value or "").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Preserve resolved profiles and geocode only new or previously unresolved profiles",
    )
    args = parser.parse_args()
    cache = ROOT / "storage/artist_location_cache"
    raw = json.loads((cache / "jamendo_artist_records.json").read_text())
    output_path = ROOT / "storage/public/merged/jamendo_artist_locations.json"
    previous = json.loads(output_path.read_text()) if args.new_only and output_path.exists() else {"artists": {}}
    preserved = previous["artists"]
    geo = json.loads(re.search(r"window.EchoGeoData = (.+);", (ROOT / "web/geo-data.js").read_text()).group(1))
    iso = {v["alpha3"]: k for k, v in geo.items()}
    pending = {}
    for aid, artist in raw["artists"].items():
        if "jamendo:" + aid in preserved:
            continue
        locations = artist.get("locations") or []
        if len(locations) != 1:
            continue
        loc = locations[0]
        code = iso.get(loc.get("country"))
        if code and clean(loc.get("city")):
            pending[code + "|" + clean(loc["city"]).casefold()] = (code, clean(loc["city"]))
    country_qids = json.loads((cache / "country_qids.json").read_text())
    queries = []
    for chunk in batches(sorted(pending), 6):
        tuples = []
        for key in chunk:
            code, city = pending[key]
            if code not in country_qids:
                continue
            variants = {city, city.title(), re.sub(r"\s*\([^)]*\)\s*$", "", city).title()}
            languages = {
                "en",
                {
                    "FR": "fr",
                    "DE": "de",
                    "IT": "it",
                    "BR": "pt",
                    "PT": "pt",
                    "RU": "ru",
                    "UA": "uk",
                    "ES": "es",
                    "PL": "pl",
                    "JP": "ja",
                    "KR": "ko",
                }.get(code, "en"),
            }
            for text in sorted(variants):
                for lang in sorted(languages):
                    tuples.append(
                        "(" + json.dumps(key) + " wd:" + country_qids[code] + " " + json.dumps(text) + "@" + lang + ")"
                    )
        if not tuples:
            continue
        labels = " ".join(
            "OPTIONAL { ?place rdfs:label ?" + lang + " FILTER(LANG(?" + lang + ')="' + lang + '") }'
            for lang in ["en", "zh", "de", "es", "pt", "ja", "ko"]
        )
        queries.append(
            'PREFIX hint: <http://www.bigdata.com/queryHints#>\nSELECT DISTINCT ?lookup ?place ?coord ?typeLabel ?en ?zh ?de ?es ?pt ?ja ?ko WHERE { hint:Query hint:optimizer "None" . VALUES (?lookup ?country ?label) { '
            + " ".join(tuples)
            + ' } ?place (rdfs:label|skos:altLabel) ?label; wdt:P17 ?country; wdt:P625 ?coord . ?place wdt:P31 ?kind . ?kind rdfs:label ?typeLabel FILTER(LANG(?typeLabel)="en") . '
            + labels
            + " }"
        )
    records = fetch_all(queries, cache, "Jamendo city geocoding")
    found = defaultdict(dict)
    for record in records:
        coords = point(value(record, "coord"))
        kind = value(record, "typeLabel").lower()
        if (
            coords
            and not ("former" in kind or "historical" in kind)
            and re.search(r"city|town|village|municipality|commune|settlement|capital|urban", kind)
        ):
            found[value(record, "lookup")][value(record, "place")] = {
                "lat": coords[0],
                "lng": coords[1],
                "city_labels": {
                    "zh-CN" if lang == "zh" else lang: value(record, lang)
                    for lang in ["en", "zh", "de", "es", "pt", "ja", "ko"]
                    if value(record, lang)
                },
                "location_place_url": value(record, "place").replace("http:", "https:"),
                "place_type": kind,
            }
    capitals = json.loads((cache / "capitals.json").read_text())
    artists = dict(preserved)
    today = datetime.now(timezone.utc).isoformat()
    for aid, artist in raw["artists"].items():
        if "jamendo:" + aid in preserved:
            continue
        locations = artist.get("locations") or []
        if len(locations) != 1:
            continue
        loc = locations[0]
        code = iso.get(loc.get("country"))
        city = clean(loc.get("city"))
        if not code:
            continue
        matches = found.get(code + "|" + city.casefold(), {})
        entry = None
        points = list(matches.values())
        clustered = (
            points
            and max(p["lat"] for p in points) - min(p["lat"] for p in points) < 0.25
            and max(p["lng"] for p in points) - min(p["lng"] for p in points) < 0.25
        )
        if len(matches) == 1 or clustered:
            selected = sorted(
                points,
                key=lambda p: (not bool(re.search(r"city|town|capital", p["place_type"])), p["location_place_url"]),
            )[0]
            entry = {
                **selected,
                "city": city,
                "location_precision": "artist_city",
                "location_relation": "public_artist_profile",
            }
        elif city and loc.get("latitude") is not None and loc.get("longitude") is not None:
            try:
                lat, lng = float(loc["latitude"]), float(loc["longitude"])
                if -90 <= lat <= 90 and -180 <= lng <= 180 and (lat, lng) != (0, 0):
                    # Display at city scale, never an individual's precise address.
                    entry = {
                        "lat": round(lat, 2),
                        "lng": round(lng, 2),
                        "city": city,
                        "city_labels": {"en": city},
                        "location_precision": "artist_city",
                        "location_relation": "public_artist_profile",
                    }
            except (TypeError, ValueError):
                pass
        if entry is None and code in capitals:
            cap = capitals[code]
            entry = {
                "lat": cap["point"][0],
                "lng": cap["point"][1],
                "city": cap["labels"].get("en", ""),
                "city_labels": cap["labels"],
                "location_precision": "country_capital",
                "location_relation": "artist_country",
                "location_place_url": cap["url"].replace("http:", "https:"),
            }
        if entry:
            artists["jamendo:" + aid] = {
                **entry,
                "artist": artist["name"],
                "country": code,
                "country_source": "artist_location",
                "location_source_url": "https://www.jamendo.com/artist/" + aid,
                "location_checked_at": today,
                "declared_city": city,
            }
    output = {"schema_version": 1, "updated_at": today, "artists": artists, "tracks": raw["tracks"]}
    atomic(ROOT / "storage/public/merged/jamendo_artist_locations.json", output)
    print(
        "Resolved Jamendo artists:",
        len(artists),
        "cities:",
        sum(a["location_precision"] == "artist_city" for a in artists.values()),
        flush=True,
    )


if __name__ == "__main__":
    main()
