"""Verify remaining artists through exact public aliases plus catalog recordings."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from enrich_artist_locations import ROOT, artist_key, atomic, batches
from search_remaining_artist_locations import CACHE, norm, phrase, search
from repair_truncated_artist_search import title_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument(
        "--lookup-snapshot",
        type=Path,
        default=ROOT / "storage/artist_location_cache/alias_search_lookups.json",
        help="Keep batch boundaries stable when resuming after more matches are cached",
    )
    args = parser.parse_args()
    base = json.loads((ROOT / "storage/public/merged/artist_locations.json").read_text())
    prior = json.loads((ROOT / "storage/public/merged/recording_verified_artist_locations.json").read_text())
    canonical = json.loads((ROOT / "storage/artist_location_cache/canonical_artist_names.json").read_text())
    groups = defaultdict(list)
    for row in csv.DictReader(
        (ROOT / "storage/public/merged/metadata_merged.csv").open(encoding="utf-8-sig", newline="")
    ):
        key = artist_key(row)
        if (
            key not in base["artists"]
            and base.get("tracks", {}).get(row["track_id"]) not in base["artists"]
            and not prior["audit"].get(key, {}).get("recording_evidence", {}).get("artist_url")
        ):
            groups[key].append(row)
    names = defaultdict(set)
    titles = {}
    for key, rows in groups.items():
        for name in {canonical.get(key, {}).get("name", "")} | {r["artist"] for r in rows}:
            if name:
                names[norm(name)].add(key)
        titles[key] = set().union(*(title_variants(r["title"]) for r in rows))
    profiles = {}
    songs = defaultdict(dict)
    for path in CACHE.glob("artist-*.json"):
        for artist in json.loads(path.read_text()).get("artists", []):
            profiles[artist["id"]] = artist

    def add_songs(recordings):
        for recording in recordings:
            for credit in recording.get("artist-credit", []):
                if isinstance(credit, dict) and credit.get("artist"):
                    mid = credit["artist"]["id"]
                    for title in title_variants(recording["title"]):
                        songs[mid][title] = recording

    for path in CACHE.glob("recording-*.json"):
        add_songs(json.loads(path.read_text()).get("recordings", []))
    candidates = defaultdict(set)

    def add_profiles(artists):
        for artist in artists:
            profiles[artist["id"]] = artist
            aliases = {artist["name"], artist.get("sort-name", "")} | {a["name"] for a in artist.get("aliases", [])}
            for alias in aliases:
                for key in names.get(norm(alias), []):
                    candidates[key].add(artist["id"])

    add_profiles(list(profiles.values()))

    def matches(key):
        return {
            mid: next(iter(titles[key] & set(songs[mid]))) for mid in candidates[key] if titles[key] & set(songs[mid])
        }

    ordered = sorted(groups, key=lambda k: -len(groups[k]))
    need = [key for key in ordered if not matches(key)]
    lookups = [(key, canonical.get(key, {}).get("name") or groups[key][0]["artist"]) for key in need]
    if args.lookup_snapshot.exists():
        lookups = json.loads(args.lookup_snapshot.read_text())
    elif not args.cached_only:
        atomic(args.lookup_snapshot, lookups)
    print(
        f"Alias pass: {len(groups)} groups; cached exact aliases {sum(bool(v) for v in candidates.values())}; names to query {len(lookups)}",
        flush=True,
    )
    if not args.cached_only:
        for i, chunk in enumerate(batches(lookups, 20), 1):
            text = " OR ".join("(artist:" + phrase(name) + " OR alias:" + phrase(name) + ")" for _, name in chunk)
            add_profiles(search("artist", text).get("artists", []))
            if i % 5 == 0:
                print(f"Artist alias lookup {i}/{len(batches(lookups, 20))}", flush=True)
        checks = []
        for key in ordered:
            if matches(key) or not candidates[key] or len(candidates[key]) > 8:
                continue
            best = []
            for row in groups[key]:
                title = re.sub(r"\s*[\(\[].*?[\)\]]", "", row["title"]).strip()
                if (
                    4 <= len(norm(title))
                    and len(title.encode()) <= 400
                    and norm(title) not in {"intro", "outro", "interlude", "untitled", "remix", "theme", "instrumental"}
                    and title not in best
                ):
                    best.append(title)
            for mid in sorted(candidates[key]):
                for title in best[:2]:
                    checks.append((mid, title))
        for i, chunk in enumerate(batches(list(dict.fromkeys(checks)), 15), 1):
            text = " OR ".join("(arid:" + mid + " AND recording:" + phrase(title) + ")" for mid, title in chunk)
            add_songs(search("recording", text).get("recordings", []))
            if i % 5 == 0:
                print(f"Alias discography verification {i}/{len(batches(list(dict.fromkeys(checks)), 15))}", flush=True)
    proposed = {}
    for key in ordered:
        verified = matches(key)
        if len(verified) != 1:
            continue
        mid, title = next(iter(verified.items()))
        recording = songs[mid][title]
        proposed[key] = {
            "artist": groups[key][0]["artist"],
            "search_name": canonical.get(key, {}).get("name") or groups[key][0]["artist"],
            "status": "recording_verified_pending_location",
            "identity_method": "exact_artist_alias_and_catalog_recording",
            "recording_evidence": {
                "artist_url": "https://musicbrainz.org/artist/" + mid,
                "recording_url": "https://musicbrainz.org/recording/" + recording["id"],
                "recording_title": recording["title"],
                "identity_method": "exact_artist_alias_and_catalog_recording",
                "matched_artist_name": profiles[mid]["name"],
            },
        }
    report = {
        "pending_groups": len(groups),
        "groups_with_exact_public_alias": sum(bool(v) for v in candidates.values()),
        "recording_verified_identities": len(proposed),
        "cached_only": args.cached_only,
    }
    atomic(
        ROOT / "storage/artist_location_cache/alias_verified_candidates.json",
        {"schema_version": 1, "report": report, "proposed": proposed},
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
