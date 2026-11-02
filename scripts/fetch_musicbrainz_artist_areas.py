"""Resolve remaining ambiguous nationalities through identifier-linked MusicBrainz areas."""

from __future__ import annotations
import hashlib
import json
import time
import urllib.request
import urllib.parse
from enrich_artist_locations import ROOT, query, atomic, value, batches, point


def main():
    cache = ROOT / "storage/artist_location_cache"
    base = json.loads((ROOT / "storage/public/merged/artist_locations.json").read_text())
    output = ROOT / "storage/public/merged/musicbrainz_artist_locations.json"
    existing = json.loads(output.read_text())["artists"] if output.exists() else {}
    pending = {
        key
        for key, record in base["unresolved"].items()
        if key.startswith("itunes:") and record["reason"] == "no_unambiguous_location"
    }
    pending.update(existing)
    pending.update(
        key
        for key, record in base["artists"].items()
        if "musicbrainz.org/artist/" in record.get("location_source_url", "")
    )
    entities = {}
    for path in cache.glob("*.json"):
        if path.name.startswith("jamendo"):
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            continue
        for row in data:
            key = "itunes:" + value(row, "apple")
            if key in pending:
                entities[key] = value(row, "artist")
    qids = sorted(set(entities.values()))
    mbids = {}
    for chunk in batches(qids, 150):
        q = (
            "SELECT ?artist ?mbid WHERE { VALUES ?artist { "
            + " ".join("wd:" + x.rsplit("/", 1)[-1] for x in chunk)
            + " } ?artist wdt:P434 ?mbid }"
        )
        for row in query(q, cache):
            mbids[value(row, "artist")] = value(row, "mbid")
    found = {}
    for i, chunk in enumerate(batches(sorted(set(mbids.values())), 25), 1):
        search = " OR ".join("arid:" + mid for mid in chunk)
        path = cache / ("musicbrainz-" + hashlib.sha256(search.encode()).hexdigest() + ".json")
        if path.exists():
            records = json.loads(path.read_text())
        else:
            url = "https://musicbrainz.org/ws/2/artist/?" + urllib.parse.urlencode(
                {"query": search, "fmt": "json", "limit": 100}
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "EchoMusicCatalog/1.0 (public artist geography research)"}
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=35) as response:
                        records = json.load(response)["artists"]
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(4 * (attempt + 1))
            atomic(path, records)
            time.sleep(1.1)
        for record in records:
            if record.get("id") in chunk:
                found[record["id"]] = record
        print(f"MusicBrainz exact IDs: {i} batches, {len(found)} artists", flush=True)
    # Search indexes can omit merged identifiers. The canonical lookup resolves
    # redirects without falling back to a potentially ambiguous artist name.
    for mid in sorted(set(mbids.values()) - set(found)):
        path = cache / ("musicbrainz-direct-" + mid + ".json")
        if path.exists():
            record = json.loads(path.read_text())
        else:
            req = urllib.request.Request(
                "https://musicbrainz.org/ws/2/artist/" + mid + "?fmt=json",
                headers={"User-Agent": "EchoMusicCatalog/1.0 (public artist geography research)"},
            )
            try:
                with urllib.request.urlopen(req, timeout=25) as response:
                    record = json.load(response)
                atomic(path, record)
            except OSError:
                time.sleep(1.1)
                continue
            time.sleep(1.1)
        if record.get("name"):
            found[mid] = record
    area_ids = {
        a.get("area", {}).get("id")
        for a in found.values()
        if a.get("area", {}).get("type") != "Country" and not a.get("area", {}).get("iso-3166-1-codes")
    } - {None}
    cities = {}
    for chunk in batches(sorted(area_ids), 80):
        q = (
            "SELECT ?mbarea ?place ?coord ?code ?en ?settlement WHERE { VALUES ?mbarea { "
            + " ".join(json.dumps(i) for i in chunk)
            + ' } ?place wdt:P982 ?mbarea;wdt:P625 ?coord. FILTER NOT EXISTS {?place wdt:P297 ?ownCountryCode} BIND(EXISTS {?place wdt:P31/wdt:P279* wd:Q486972} AS ?settlement) OPTIONAL {?place wdt:P17/wdt:P297 ?code} OPTIONAL {?place rdfs:label ?en FILTER(LANG(?en)="en")} }'
        )
        for row in query(q, cache):
            p = point(value(row, "coord"))
            if p and value(row, "settlement") == "true":
                cities[value(row, "mbarea")] = {
                    "point": p,
                    "country": value(row, "code"),
                    "city": value(row, "en"),
                    "url": value(row, "place"),
                }
    capitals = json.loads((cache / "capitals.json").read_text())
    area_countries = {}
    all_areas = sorted({a.get("area", {}).get("id") for a in found.values()} - {None})
    for chunk in batches(all_areas, 80):
        q = (
            "SELECT ?mbarea ?code WHERE { VALUES ?mbarea { "
            + " ".join(json.dumps(i) for i in chunk)
            + " } ?place wdt:P982 ?mbarea. ?place (wdt:P297|wdt:P17/wdt:P297) ?code }"
        )
        for row in query(q, cache):
            area_countries.setdefault(value(row, "mbarea"), set()).add(value(row, "code"))
    result = dict(existing)
    for key, entity in entities.items():
        mbid = mbids.get(entity)
        artist = found.get(mbid)
        if not artist:
            continue
        result.pop(key, None)
        country = artist.get("country", "")
        area = artist.get("area") or {}
        city = cities.get(area.get("id"))
        codes = area_countries.get(area.get("id"), set())
        if not country and len(codes) == 1:
            country = next(iter(codes))
        record = None
        if city and city["country"]:
            record = {
                "country": city["country"],
                "city": city["city"],
                "city_labels": {"en": city["city"]},
                "lat": city["point"][0],
                "lng": city["point"][1],
                "location_precision": "artist_city",
                "location_relation": "musicbrainz_artist_area",
                "location_place_url": city["url"],
            }
        elif country in capitals:
            cap = capitals[country]
            record = {
                "country": country,
                "city": cap["labels"].get("en", ""),
                "city_labels": cap["labels"],
                "lat": cap["point"][0],
                "lng": cap["point"][1],
                "location_precision": "country_capital",
                "location_relation": "musicbrainz_artist_area",
                "location_place_url": cap["url"],
            }
        if record:
            result[key] = {
                **record,
                "artist": artist["name"],
                "artist_id": key[7:],
                "country_source": "artist_location",
                "location_source_url": "https://musicbrainz.org/artist/" + mbid,
                "identity_source_url": entity.replace("http:", "https:"),
            }
    atomic(output, {"artists": result})
    print("MusicBrainz additional located artists:", len(result), flush=True)


if __name__ == "__main__":
    main()
