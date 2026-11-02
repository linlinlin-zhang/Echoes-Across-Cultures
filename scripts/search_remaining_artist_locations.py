"""Search unresolved artists by exact names/aliases and verify against catalog recordings.

Uses only public MusicBrainz and Wikidata requests. Results are cached; no LLM or
translation API calls. A matching name alone never authorizes geographic replacement.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import http.client
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from enrich_artist_locations import ROOT, query, atomic, value, batches, point, artist_key

CACHE = ROOT / "storage/artist_location_cache/name_search"
LAST = 0.0


def norm(text):
    text = unicodedata.normalize("NFKD", str(text or "")).casefold()
    return "".join(c for c in text if not unicodedata.combining(c) and c.isalnum())


def phrase(text):
    return '"' + re.sub(r'([\\"+\-!(){}\[\]^~*?:/])', r"\\\1", str(text)) + '"'


def search(resource, text):
    global LAST
    path = CACHE / (resource + "-" + hashlib.sha256(text.encode()).hexdigest() + ".json")
    if path.exists():
        cached = json.loads(path.read_text())
        if int(cached.get("count", 0)) <= 100 or ") OR (" not in text or cached.get("split_complete"):
            return cached
        # A full page is not a complete result set. Split independently matched
        # clauses so a prolific artist cannot crowd other artists out of a batch.
        clauses = text[1:-1].split(") OR (")
        middle = len(clauses) // 2
        a = search(resource, "(" + ") OR (".join(clauses[:middle]) + ")")
        b = search(resource, "(" + ") OR (".join(clauses[middle:]) + ")")
        merged = {
            resource + "s": list({r["id"]: r for r in a.get(resource + "s", []) + b.get(resource + "s", [])}.values()),
            "count": int(a.get("count", 0)) + int(b.get("count", 0)),
            "split_complete": True,
        }
        atomic(path, merged)
        return merged
    import urllib.parse

    url = (
        "https://musicbrainz.org/ws/2/"
        + resource
        + "/?"
        + urllib.parse.urlencode({"query": text, "fmt": "json", "limit": 100})
    )
    if len(url) > 7000 and ") OR (" in text:
        clauses = text[1:-1].split(") OR (")
        middle = len(clauses) // 2
        a = search(resource, "(" + ") OR (".join(clauses[:middle]) + ")")
        b = search(resource, "(" + ") OR (".join(clauses[middle:]) + ")")
        merged = {
            resource + "s": list({r["id"]: r for r in a.get(resource + "s", []) + b.get(resource + "s", [])}.values()),
            "count": int(a.get("count", 0)) + int(b.get("count", 0)),
            "split_complete": True,
        }
        atomic(path, merged)
        return merged
    for attempt in range(5):
        time.sleep(max(0, 1.1 - (time.monotonic() - LAST)))
        LAST = time.monotonic()
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "EchoMusicCatalog/1.0 (https://resonance.website; public artist location and recording verification)"
                },
            )
            with urllib.request.urlopen(req, timeout=35) as response:
                data = json.load(response)
            atomic(path, data)
            if int(data.get("count", 0)) > 100 and ") OR (" in text:
                return search(resource, text)
            return data
        except (OSError, http.client.HTTPException, ValueError) as error:
            if attempt == 4:
                raise
            detail = f" HTTP {error.code}" if hasattr(error, "code") else ""
            print(f"MusicBrainz retry {attempt + 1}/4: {type(error).__name__}{detail}", flush=True)
            time.sleep(4 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Artist groups; 0 means all remaining")
    parser.add_argument("--retry-checked", action="store_true", help="Revisit previously audited unresolved artists")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "storage/public/merged/artist_locations.json",
        help="Catalog location snapshot used to select unresolved identities",
    )
    parser.add_argument(
        "--locations-only",
        action="store_true",
        help="Rebuild geography for saved verified identities using cached artist responses",
    )
    parser.add_argument(
        "--identity-input",
        type=Path,
        help="Additional identities verified against public aliases and actual catalog recordings",
    )
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    rows = list(
        csv.DictReader((ROOT / "storage/public/merged/metadata_merged.csv").open(encoding="utf-8-sig", newline=""))
    )
    groups = defaultdict(list)
    for row in rows:
        if (
            artist_key(row) not in base["artists"]
            and base.get("tracks", {}).get(row["track_id"]) not in base["artists"]
        ):
            groups[artist_key(row)].append(row)
    output = ROOT / "storage/public/merged/recording_verified_artist_locations.json"
    prior = json.loads(output.read_text()) if output.exists() else {"artists": {}, "audit": {}}
    if args.identity_input:
        for key, audit in json.loads(args.identity_input.read_text())["proposed"].items():
            if key in groups and re.fullmatch(
                r"https://musicbrainz.org/artist/[0-9a-f-]{36}",
                audit.get("recording_evidence", {}).get("artist_url", ""),
            ):
                prior["audit"][key] = audit
    canonical_path = ROOT / "storage/artist_location_cache/canonical_artist_names.json"
    canonical = json.loads(canonical_path.read_text()) if canonical_path.exists() else {}

    def should_search(key):
        audit = prior["audit"].get(key)
        return (
            args.retry_checked
            or not audit
            or (
                key in canonical
                and norm(canonical[key]["name"]) != norm(audit.get("search_name") or audit.get("artist"))
            )
        )

    ordered = sorted((k for k in groups if should_search(k)), key=lambda k: -len(groups[k]))
    if args.locations_only:
        ordered = sorted(
            (
                k
                for k, a in prior["audit"].items()
                if k in groups
                and re.fullmatch(
                    r"https://musicbrainz.org/artist/[0-9a-f-]{36}",
                    a.get("recording_evidence", {}).get("artist_url", ""),
                )
            ),
            key=lambda k: -len(groups[k]),
        )
    if args.limit:
        ordered = ordered[: args.limit]
    names = defaultdict(set)
    for key in ordered:
        if key in canonical:
            names[canonical[key]["name"]].add(key)
        else:
            for row in groups[key]:
                names[row["artist"].strip()].add(key)
    # A recording search can establish both the artist identity and discography
    # match at once; avoid thousands of separate fuzzy artist-name requests.
    checks = []
    for key in [] if args.locations_only else ordered:
        search_names = (
            [canonical[key]["name"]] if key in canonical else list(dict.fromkeys(r["artist"] for r in groups[key]))[:2]
        )
        for name in search_names:
            for title in list(dict.fromkeys(r["title"] for r in groups[key]))[:2]:
                title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip()
                if (
                    4 <= len(norm(title))
                    and len(title.encode("utf-8")) <= 500
                    and norm(title)
                    not in {
                        "intro",
                        "outro",
                        "interlude",
                        "untitled",
                        "track1",
                        "track01",
                        "originalmix",
                        "remix",
                        "theme",
                        "instrumental",
                        "improvisation",
                        "live",
                    }
                ):
                    checks.append((key, name, title))
    print(f"Remaining artist groups: {len(ordered)}; recording checks: {len(checks)}", flush=True)
    verified = defaultdict(dict)
    if args.locations_only:
        for key in ordered:
            evidence = prior["audit"][key]["recording_evidence"]
            verified[key][evidence["artist_url"].rsplit("/", 1)[1]] = evidence
    candidates = defaultdict(set)
    for i, chunk in enumerate(batches(checks, 15), 1):
        text = " OR ".join(
            "(artist:" + phrase(name) + " AND recording:" + phrase(title) + ")" for _, name, title in chunk
        )
        data = search("recording", text)
        for recording in data.get("recordings", []):
            for credit in recording.get("artist-credit", []):
                if not isinstance(credit, dict) or not credit.get("artist"):
                    continue
                artist = credit["artist"]
                credit_names = {norm(artist.get("name")), norm(credit.get("name"))}
                for key, name, title in chunk:
                    if norm(name) in credit_names and norm(recording.get("title")) == norm(title):
                        candidates[key].add(artist["id"])
                        verified[key][artist["id"]] = {
                            "recording_title": recording["title"],
                            "recording_url": "https://musicbrainz.org/recording/" + recording["id"],
                            "artist_url": "https://musicbrainz.org/artist/" + artist["id"],
                        }
        if i % 10 == 0:
            print(
                f"Recording verification {i}/{len(batches(checks, 15))}: {len(verified)} supported artist groups",
                flush=True,
            )
    accepted = {key: next(iter(matches)) for key, matches in verified.items() if len(matches) == 1}
    records = {}
    if args.locations_only:
        wanted = set(accepted.values())
        for path in sorted(CACHE.glob("artist-*.json"), key=lambda p: p.stat().st_mtime):
            for artist in json.loads(path.read_text()).get("artists", []):
                if artist["id"] in wanted:
                    records[artist["id"]] = artist
        for chunk in batches(sorted(wanted - set(records)), 100):
            data = search("artist", " OR ".join("arid:" + mid for mid in chunk))
            for artist in data.get("artists", []):
                if artist["id"] in wanted:
                    records[artist["id"]] = artist
    artist_batches = batches(sorted(set(accepted.values())), 25)
    for i, chunk in enumerate([] if args.locations_only else artist_batches, 1):
        data = search("artist", " OR ".join("arid:" + mid for mid in chunk))
        for artist in data.get("artists", []):
            if artist["id"] in chunk:
                records[artist["id"]] = artist
        if i % 10 == 0:
            print(f"Artist areas {i}/{len(artist_batches)}", flush=True)
    accepted = {key: mid for key, mid in accepted.items() if mid in records}
    # Artist IDs found via recording evidence can reveal cities which the generic
    # MusicBrainz area omits (notably a band's formation city).
    artist_cities = defaultdict(lambda: defaultdict(dict))
    cache = ROOT / "storage/artist_location_cache"
    city_batches = batches(sorted(set(accepted.values())), 50)
    for i, chunk in enumerate(city_batches, 1):
        text = (
            'SELECT ?mid ?artist ?place ?kind ?coord ?code ?en ?settlement WHERE { hint:Query hint:optimizer "None". VALUES ?mid { '
            + " ".join(json.dumps(mid) for mid in chunk)
            + ' } ?artist wdt:P434 ?mid. { ?artist wdt:P937 ?place BIND("work" AS ?kind) } UNION { ?artist wdt:P740 ?place BIND("formation" AS ?kind) } ?place wdt:P625 ?coord;wdt:P17/wdt:P297 ?code. FILTER NOT EXISTS {?place wdt:P297 ?ownCode} BIND(EXISTS {?place wdt:P31/wdt:P279* wd:Q486972} AS ?settlement) OPTIONAL {?place rdfs:label ?en FILTER(LANG(?en)="en")} }'
        )
        for row in query(text, cache):
            coords = point(value(row, "coord"))
            if coords:
                artist_cities[value(row, "mid")][value(row, "kind")][value(row, "place")] = {
                    "country": value(row, "code"),
                    "city": value(row, "en"),
                    "lat": coords[0],
                    "lng": coords[1],
                    "location_precision": "artist_city",
                    "location_place_url": value(row, "place"),
                    "city_source_url": value(row, "artist"),
                    "location_relation": "work_location" if value(row, "kind") == "work" else "formation_location",
                }
        print(f"Artist city lookup {i}/{len(city_batches)}", flush=True)
    area_ids = sorted({records[mid].get("area", {}).get("id") for mid in accepted.values()} - {None})
    geography = {}
    area_batches = batches(area_ids, 80)
    for i, chunk in enumerate(area_batches, 1):
        text = (
            'SELECT ?mbarea ?place ?coord ?code ?ownCode ?en ?settlement WHERE { hint:Query hint:optimizer "None". VALUES ?mbarea { '
            + " ".join(json.dumps(mid) for mid in chunk)
            + ' } ?place wdt:P982 ?mbarea. OPTIONAL {?place wdt:P625 ?coord} OPTIONAL {?place wdt:P297 ?ownCode} OPTIONAL {?place wdt:P17/wdt:P297 ?code} BIND(EXISTS {?place wdt:P31/wdt:P279* wd:Q486972} AS ?settlement) OPTIONAL {?place rdfs:label ?en FILTER(LANG(?en)="en")} }'
        )
        for row in query(text, cache):
            geography[value(row, "mbarea")] = {
                "point": point(value(row, "coord")),
                "country": value(row, "ownCode") or value(row, "code"),
                "city": value(row, "en"),
                "own_country_code": value(row, "ownCode"),
                "url": value(row, "place"),
            }
        print(f"Area geography {i}/{len(area_batches)}", flush=True)
    # Blazegraph can omit BIND(EXISTS(...)) with its optimizer disabled. Verify
    # city types explicitly in an anchored query instead of treating omission
    # as a negative result and silently downgrading every city to a capital.
    places = {
        loc["location_place_url"]
        for kinds in artist_cities.values()
        for entries in kinds.values()
        for loc in entries.values()
    }
    places.update(loc["url"] for loc in geography.values())
    settlements = set()
    for i, chunk in enumerate(batches(sorted(places), 80), 1):
        ids = [url.rsplit("/", 1)[1] for url in chunk if re.fullmatch(r"https?://www.wikidata.org/entity/Q\d+", url)]
        text = (
            'SELECT DISTINCT ?place WHERE { hint:Query hint:optimizer "None". VALUES ?place { '
            + " ".join("wd:" + qid for qid in ids)
            + " } ?place wdt:P31/wdt:P279* wd:Q486972. }"
        )
        settlements.update(value(row, "place") for row in query(text, cache))
        print(f"City type verification {i}/{len(batches(sorted(places), 80))}", flush=True)
    for kinds in artist_cities.values():
        for kind, entries in list(kinds.items()):
            kinds[kind] = {url: loc for url, loc in entries.items() if url in settlements}
    for loc in geography.values():
        loc["is_city"] = loc["url"] in settlements and not loc["own_country_code"]
    # A verified MusicBrainz artist may omit its country while its linked
    # Wikidata biography explicitly provides one. Accept a unique country only.
    country_candidates = defaultdict(dict)
    missing_country = sorted(
        {
            mid
            for mid in accepted.values()
            if not records[mid].get("country")
            and not geography.get(records[mid].get("area", {}).get("id"), {}).get("country")
        }
    )
    for i, chunk in enumerate(batches(missing_country, 50), 1):
        text = (
            "SELECT DISTINCT ?mid ?artist ?code0 ?code1 ?code2 WHERE { VALUES ?mid { "
            + " ".join(json.dumps(mid) for mid in chunk)
            + " } ?artist wdt:P434 ?mid. OPTIONAL { ?artist wdt:P27 ?nation0. ?nation0 wdt:P297 ?code0. } OPTIONAL { ?artist wdt:P495 ?nation1. ?nation1 wdt:P297 ?code1. } OPTIONAL { ?artist wdt:P17 ?nation2. ?nation2 wdt:P297 ?code2. } }"
        )
        for row in query(text, cache):
            for field in ["code0", "code1", "code2"]:
                if value(row, field):
                    country_candidates[value(row, "mid")][value(row, field)] = value(row, "artist")
        print(f"Biography countries {i}/{len(batches(missing_country, 50))}", flush=True)
    capitals = json.loads((cache / "capitals.json").read_text())
    for key in ordered:
        mid = accepted.get(key)
        if not mid:
            prior["audit"][key] = {
                "artist": groups[key][0]["artist"],
                "status": "ambiguous_recording_match" if len(verified[key]) > 1 else "no_confirmed_recording_match",
                "candidate_count": len(candidates[key]),
            }
            continue
        artist = records[mid]
        area = artist.get("area") or {}
        location = geography.get(area.get("id"), {})
        country = location.get("country") or artist.get("country")
        entry = None
        country_evidence = ""
        if not country and len(country_candidates[mid]) == 1:
            country, country_evidence = next(iter(country_candidates[mid].items()))
        for kind in ["work", "formation"]:
            if len(artist_cities[mid][kind]) == 1:
                entry = dict(next(iter(artist_cities[mid][kind].values())))
                break
        if entry:
            pass
        elif location.get("is_city") and location.get("point") and country:
            entry = {
                "country": country,
                "city": location["city"],
                "city_labels": {"en": location["city"]},
                "lat": location["point"][0],
                "lng": location["point"][1],
                "location_precision": "artist_city",
                "location_place_url": location["url"],
            }
        elif country in capitals:
            cap = capitals[country]
            entry = {
                "country": country,
                "city": cap["labels"].get("en", ""),
                "city_labels": cap["labels"],
                "lat": cap["point"][0],
                "lng": cap["point"][1],
                "location_precision": "country_capital",
                "location_place_url": cap["url"],
            }
        if entry:
            prior["artists"][key] = {
                **entry,
                "artist": artist["name"],
                "artist_id": key[7:] if key.startswith("itunes:") else "",
                "location_relation": entry.get("location_relation", "recording_verified_musicbrainz_area"),
                "country_source": "artist_location",
                "location_source_url": "https://musicbrainz.org/artist/" + mid,
                "identity_evidence": verified[key][mid],
            }
            if entry.get("city_source_url"):
                prior["artists"][key]["location_source_url"] = (
                    entry["city_source_url"].replace("http:", "https:").replace("/entity/", "/wiki/")
                )
            elif country_evidence:
                prior["artists"][key]["location_source_url"] = country_evidence.replace("http:", "https:").replace(
                    "/entity/", "/wiki/"
                )
                prior["artists"][key]["location_relation"] = "artist_country"
            prior["audit"][key] = {
                "artist": groups[key][0]["artist"],
                "status": "verified",
                "recording_evidence": verified[key][mid],
            }
        else:
            prior["audit"][key] = {
                "artist": groups[key][0]["artist"],
                "status": "recording_verified_location_unknown",
                "recording_evidence": verified[key][mid],
            }
    for key in ordered:
        if key in prior["audit"]:
            prior["audit"][key]["search_name"] = canonical.get(key, {}).get("name") or groups[key][0]["artist"]
        if key in prior["artists"] and key in canonical:
            prior["artists"][key]["official_identity_source_url"] = canonical[key]["source_url"]
    atomic(output, prior)
    print(
        json.dumps(
            {
                "searched_artist_groups": len(ordered),
                "matched_recordings": len(accepted),
                "located_artist_records_total": len(prior["artists"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
