"""Resolve artist geography from public Wikidata identifiers, with source links and caches.

No model-generated locations. Apple Music IDs match directly; name-only artists require
an exact label/alias match to a unique entity carrying a MusicBrainz artist identifier.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PREFIX = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
"""
FIELDS = """OPTIONAL { ?artist rdfs:label ?name FILTER(LANG(?name)="en") }
OPTIONAL { ?artist (wdt:P27|wdt:P495|wdt:P17) ?country . ?country wdt:P297 ?countryCode }
OPTIONAL { ?artist wdt:P937 ?work }
OPTIONAL { ?artist wdt:P740 ?formation }
"""


def normalized(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


def artist_key(row):
    return (
        "itunes:" + row["artist_id"].strip()
        if row.get("artist_id", "").strip()
        else "name:" + normalized(row["artist"])
    )


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(0.2)
    temp.replace(path)


def query(text, cache):
    legacy_text = PREFIX.split("PREFIX p:")[0] + text
    text = PREFIX + text
    target = cache / (hashlib.sha256(text.encode()).hexdigest() + ".json")
    if target.exists():
        return json.loads(target.read_text())
    legacy = cache / (hashlib.sha256(legacy_text.encode()).hexdigest() + ".json")
    if legacy.exists():
        return json.loads(legacy.read_text())
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": text, "format": "json"})
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": "EchoMusicCatalog/1.0 (public artist geography research)",
                },
            )
            with urllib.request.urlopen(req, timeout=55) as response:
                data = json.load(response)["results"]["bindings"]
            atomic(target, data)
            time.sleep(0.4)
            return data
        except (OSError, http.client.HTTPException, ValueError, KeyError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def batches(values, size):
    return [values[i : i + size] for i in range(0, len(values), size)]


def fetch_all(queries, cache, label):
    records = []
    with ThreadPoolExecutor(2) as pool:
        remaining = iter(queries)
        active = {pool.submit(query, q, cache) for q in [next(remaining, None), next(remaining, None)] if q}
        count = 0
        while active:
            future = next(as_completed(active))
            active.remove(future)
            records.extend(future.result())
            count += 1
            print(f"{label}: {count}/{len(queries)} batches, {len(records)} rows", flush=True)
            q = next(remaining, None)
            if q:
                active.add(pool.submit(query, q, cache))
    return records


def value(row, key):
    return row.get(key, {}).get("value", "")


def qid(url):
    result = str(url).rsplit("/", 1)[-1]
    if not re.fullmatch(r"Q\d+", result):
        raise ValueError("Invalid Wikidata entity")
    return result


def point(value):
    match = re.fullmatch(r"Point\((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)", value)
    if match:
        lng, lat = map(float, match.groups())
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "storage/public/merged/metadata_merged.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "storage/public/merged/artist_locations.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "storage/artist_location_cache")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.metadata.open(encoding="utf-8-sig", newline="")))
    groups = defaultdict(list)
    for row in rows:
        if row.get("artist", "").strip():
            groups[artist_key(row)].append(row)
    print(f"Catalog: {len(rows)} tracks, {len(groups)} artist identities", flush=True)
    ids = sorted((k[7:] for k in groups if k.startswith("itunes:")), key=lambda i: -len(groups["itunes:" + i]))
    queries = []
    for chunk in batches(ids, 100):
        vals = " ".join(json.dumps(i) for i in chunk)
        queries.append(
            "SELECT ?artist ?apple ?name ?country ?countryCode ?work ?formation WHERE { VALUES ?apple { "
            + vals
            + " } ?artist wdt:P2850 ?apple . "
            + FIELDS
            + " }"
        )
    raw = fetch_all(queries, args.cache, "Apple ID lookup")
    names = {group[0]["artist"].strip() for key, group in groups.items() if key.startswith("name:")}
    # Exact name and explicit musician identifier only; collisions are not auto-selected.
    queries = []
    for chunk in batches(sorted(names), 35):
        vals = " ".join("(" + json.dumps(n) + " " + json.dumps(n) + "@en)" for n in chunk)
        queries.append(
            "SELECT ?artist ?input ?name ?country ?countryCode ?work ?formation WHERE { VALUES (?input ?label) { "
            + vals
            + " } ?artist (rdfs:label|skos:altLabel) ?label . ?artist wdt:P434 ?mbid . "
            + FIELDS
            + " }"
        )
    raw.extend(fetch_all(queries, args.cache, "Exact musician name lookup"))
    entities = defaultdict(lambda: defaultdict(set))
    candidates = defaultdict(set)
    for record in raw:
        entity = value(record, "artist")
        key = (
            "itunes:" + value(record, "apple")
            if value(record, "apple")
            else "name:" + normalized(value(record, "input"))
        )
        candidates[key].add(entity)
        for field in ["countryCode", "work", "formation", "name"]:
            entry = value(record, field)
            if field in ("work", "formation") and not re.search(r"/Q\d+$", entry):
                continue
            if entry:
                entities[entity][field].add(entry)
    places = sorted({p for e in entities.values() for key in ["work", "formation"] for p in e[key]})
    queries = []
    for chunk in batches(places, 80):
        vals = " ".join("wd:" + qid(p) for p in chunk)
        labels = " ".join(
            "OPTIONAL { ?place rdfs:label ?" + key + " FILTER(LANG(?" + key + ')="' + lang + '") }'
            for key, lang in [
                ("en", "en"),
                ("zh", "zh"),
                ("de", "de"),
                ("es", "es"),
                ("pt", "pt"),
                ("ja", "ja"),
                ("ko", "ko"),
            ]
        )
        queries.append(
            "SELECT ?place ?coord ?code ?settlement ?en ?zh ?de ?es ?pt ?ja ?ko WHERE { VALUES ?place { "
            + vals
            + " } BIND(EXISTS { ?place wdt:P31/wdt:P279* wd:Q486972 } AS ?settlement) OPTIONAL { ?place wdt:P625 ?coord } OPTIONAL { ?place (wdt:P17|wdt:P131/wdt:P17) ?country . ?country wdt:P297 ?code } "
            + labels
            + " }"
        )
    place_records = fetch_all(queries, args.cache, "City coordinates")
    place_info = defaultdict(lambda: {"countries": set(), "points": set(), "labels": {}, "settlement": False})
    for record in place_records:
        info = place_info[value(record, "place")]
        info["settlement"] = info["settlement"] or value(record, "settlement") == "true"
        if value(record, "code"):
            info["countries"].add(value(record, "code"))
        coords = point(value(record, "coord"))
        if coords:
            info["points"].add(coords)
        for lang in ["en", "zh", "de", "es", "pt", "ja", "ko"]:
            if value(record, lang):
                info["labels"]["zh-CN" if lang == "zh" else lang] = value(record, lang)
    capital_query = (
        """SELECT ?country ?code ?capital ?coord ?en ?zh ?de ?es ?pt ?ja ?ko WHERE {
?country wdt:P297 ?code; p:P36 ?statement . ?statement ps:P36 ?capital .
FILTER NOT EXISTS { ?statement pq:P582 ?end FILTER(?end < NOW()) }
?capital wdt:P625 ?coord .
"""
        + " ".join(
            "OPTIONAL { ?capital rdfs:label ?" + lang + " FILTER(LANG(?" + lang + ')="' + lang + '") }'
            for lang in ["en", "zh", "de", "es", "pt", "ja", "ko"]
        )
        + " }"
    )
    capital_records = query(capital_query, args.cache)
    capitals = {}
    for record in sorted(capital_records, key=lambda r: (value(r, "code"), value(r, "en"))):
        coords = point(value(record, "coord"))
        code = value(record, "code")
        if coords and code not in capitals:
            capitals[code] = {
                "point": coords,
                "labels": {
                    "zh-CN" if lang == "zh" else lang: value(record, lang)
                    for lang in ["en", "zh", "de", "es", "pt", "ja", "ko"]
                    if value(record, lang)
                },
                "url": value(record, "capital"),
            }
    atomic(args.cache / "capitals.json", capitals)
    country_qids = {
        value(r, "code"): qid(value(r, "country"))
        for r in capital_records
        if value(r, "code") in capitals and value(r, "capital") == capitals[value(r, "code")]["url"]
    }
    atomic(args.cache / "country_qids.json", country_qids)
    result = {}
    unresolved = {}
    today = datetime.now(timezone.utc).isoformat()
    for key, group in groups.items():
        matches = candidates[key]
        if len(matches) != 1:
            unresolved[key] = {
                "artist": group[0]["artist"],
                "reason": "ambiguous_identity" if matches else "no_exact_public_identity",
            }
            continue
        entity = next(iter(matches))
        artist = entities[entity]
        location = None
        for kind in ["work", "formation"]:
            # Several cities can reflect tours or historical workplaces. Do not choose one arbitrarily.
            if len(artist[kind]) == 1:
                place = next(iter(artist[kind]))
                info = place_info[place]
                codes = info["countries"] or artist["countryCode"]
                if info["settlement"] and len(info["points"]) == 1 and len(codes) == 1 and len(next(iter(codes))) == 2:
                    location = {
                        "country": next(iter(codes)),
                        "city": info["labels"].get("en", ""),
                        "city_labels": info["labels"],
                        "lat": next(iter(info["points"]))[0],
                        "lng": next(iter(info["points"]))[1],
                        "location_precision": "artist_city",
                        "location_relation": "work_location" if kind == "work" else "formation_location",
                        "location_place_url": place,
                    }
                    break
        if location is None and len(artist["countryCode"]) == 1:
            code = next(iter(artist["countryCode"]))
            capital = capitals.get(code)
            if capital:
                location = {
                    "country": code,
                    "city": capital["labels"].get("en", ""),
                    "city_labels": capital["labels"],
                    "lat": capital["point"][0],
                    "lng": capital["point"][1],
                    "location_precision": "country_capital",
                    "location_relation": "artist_country",
                    "location_place_url": capital["url"],
                }
        if location:
            result[key] = {
                **location,
                "artist": group[0]["artist"],
                "artist_id": group[0].get("artist_id", ""),
                "location_source_url": entity.replace("http:", "https:"),
                "location_checked_at": today,
                "country_source": "artist_location",
            }
        else:
            unresolved[key] = {"artist": group[0]["artist"], "reason": "no_unambiguous_location"}
    # Reviewed public artist bios can supplement database gaps without changing identity rules.
    manual = ROOT / "configs/artist_locations.reviewed.json"
    if manual.exists():
        for key, record in json.loads(manual.read_text()).items():
            if key in groups:
                assert record.get("location_source_url", "").startswith("https://")
                assert -90 <= record["lat"] <= 90 and -180 <= record["lng"] <= 180
                result[key] = {**record, "location_checked_at": today, "country_source": "artist_location"}
                unresolved.pop(key, None)
    track_mapping = {}
    jamendo = ROOT / "storage/public/merged/jamendo_artist_locations.json"
    if jamendo.exists():
        supplement = json.loads(jamendo.read_text())
        result.update(supplement["artists"])
        track_mapping = supplement["tracks"]
        for key, group in groups.items():
            if all(track_mapping.get(row["track_id"]) in result for row in group):
                unresolved.pop(key, None)
            elif (
                key.startswith("name:")
                and key not in result
                and all(row.get("source_dataset") == "jamendo" for row in group)
            ):
                linked = {
                    track_mapping[row["track_id"]] for row in group if track_mapping.get(row["track_id"]) in result
                }
                if len(linked) == 1:
                    profile_key = next(iter(linked))
                    profile = result[profile_key]
                    if normalized(profile.get("artist")) == key[5:]:
                        result[key] = {
                            **profile,
                            "location_identity_match": "exact_credit_and_verified_jamendo_tracks",
                            "matched_artist_key": profile_key,
                            "location_checked_at": today,
                        }
                        unresolved.pop(key, None)
    musicbrainz = ROOT / "storage/public/merged/musicbrainz_artist_locations.json"
    if musicbrainz.exists():
        for key, record in json.loads(musicbrainz.read_text())["artists"].items():
            if key not in result:
                result[key] = {**record, "location_checked_at": today}
                unresolved.pop(key, None)
    recording_verified = ROOT / "storage/public/merged/recording_verified_artist_locations.json"
    recording_audit = {}
    if recording_verified.exists():
        recording_data = json.loads(recording_verified.read_text())
        recording_audit = recording_data.get("audit", {})
        for key, record in recording_data["artists"].items():
            if key not in result:
                result[key] = {**record, "location_checked_at": today}
                unresolved.pop(key, None)
    for key, record in unresolved.items():
        status = recording_audit.get(key, {}).get("status")
        if status == "recording_verified_location_unknown":
            record["reason"] = "no_public_location_for_verified_identity"
        elif status == "ambiguous_recording_match":
            record["reason"] = "ambiguous_recording_match"
    label_index_path = args.cache / "city-labels.index.json"
    label_index = json.loads(label_index_path.read_text()) if label_index_path.exists() else {}
    # Public web URLs are easier to inspect than entity RDF endpoints.
    for record in result.values():
        for field in ("location_source_url", "location_place_url"):
            if field in record:
                record[field] = (
                    record[field]
                    .replace("http://", "https://")
                    .replace("www.wikidata.org/entity/", "www.wikidata.org/wiki/")
                )
        place = record.get("location_place_url", "").rsplit("/", 1)[-1]
        if place in label_index:
            record["city_labels"] = {**record.get("city_labels", {}), **label_index[place]}
            if not record.get("city") and label_index[place].get("en"):
                record["city"] = label_index[place]["en"]
    counts = Counter(record["location_precision"] for record in result.values())
    track_precisions = Counter()
    for row in rows:
        record = result.get(track_mapping.get(row["track_id"])) or result.get(artist_key(row))
        if record:
            track_precisions[record["location_precision"]] += 1
    report = {
        "artist_identities": len(groups),
        "located_artist_records": len(result),
        "fully_located_catalog_artist_groups": len(groups) - len(unresolved),
        "unresolved_catalog_artist_groups": len(unresolved),
        "tracks_with_artist_locations": sum(track_precisions.values()),
        "precision_counts": dict(counts),
        "track_precision_counts": dict(track_precisions),
        "tracks_without_verified_artist_location": len(rows) - sum(track_precisions.values()),
    }
    atomic(
        args.output,
        {
            "schema_version": 1,
            "updated_at": today,
            "method": "Public artist identifiers, recording-verified MusicBrainz identities, Jamendo artist profiles and reviewed public biographies; no market/culture inference",
            "report": report,
            "artists": result,
            "tracks": track_mapping,
            "unresolved": unresolved,
        },
    )
    print(json.dumps(report), flush=True)
    atomic(args.output.with_suffix(".report.json"), report)
    # Keep an inspectable queue of genuinely unlocated tracks for later research.
    canonical_path = args.cache / "canonical_artist_names.json"
    canonical = json.loads(canonical_path.read_text()) if canonical_path.exists() else {}
    with args.output.with_suffix(".unresolved.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "artist_key",
                "artist",
                "canonical_artist",
                "remaining_tracks",
                "lookup_status",
                "reason",
                "example_titles",
            ],
        )
        writer.writeheader()
        missing_groups = {
            key: [row for row in groups[key] if not (result.get(track_mapping.get(row["track_id"])) or result.get(key))]
            for key in unresolved
        }
        for key in sorted(unresolved, key=lambda k: -len(missing_groups[k])):
            missing = missing_groups[key]
            if missing:
                writer.writerow(
                    {
                        "artist_key": key,
                        "artist": unresolved[key]["artist"],
                        "canonical_artist": canonical.get(key, {}).get("name", ""),
                        "remaining_tracks": len(missing),
                        "lookup_status": recording_audit.get(key, {}).get("status", "not_recording_verified"),
                        "reason": unresolved[key]["reason"],
                        "example_titles": " | ".join(dict.fromkeys(row["title"] for row in missing[:3])),
                    }
                )


if __name__ == "__main__":
    main()
