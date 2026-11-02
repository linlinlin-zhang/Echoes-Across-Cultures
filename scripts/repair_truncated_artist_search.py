"""Recover truncated public recording searches and reuse all cached discography.

Only exact artist credits and actual catalog song titles establish an identity.
Existing reviewed locations are preserved. Run search_remaining_artist_locations
with --locations-only afterwards to resolve the newly verified artist identities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict

from enrich_artist_locations import ROOT, artist_key, atomic
from search_remaining_artist_locations import CACHE, norm, phrase, search


def title_variants(title):
    return {norm(title), norm(re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip())} - {""}


def main():
    base = json.loads((ROOT / "storage/public/merged/artist_locations.json").read_text())
    snapshot = json.loads((ROOT / "tmp/artist-locations/recording-search-base.json").read_text())
    canonical = json.loads((ROOT / "storage/artist_location_cache/canonical_artist_names.json").read_text())
    rows = list(
        csv.DictReader((ROOT / "storage/public/merged/metadata_merged.csv").open(encoding="utf-8-sig", newline=""))
    )
    groups = defaultdict(list)
    old_groups = defaultdict(list)
    for row in rows:
        key = artist_key(row)
        if key not in base["artists"] and base.get("tracks", {}).get(row["track_id"]) not in base["artists"]:
            groups[key].append(row)
        if (
            key not in snapshot["artists"]
            and snapshot.get("tracks", {}).get(row["track_id"]) not in snapshot["artists"]
        ):
            old_groups[key].append(row)
    names = defaultdict(set)
    titles = {}
    for key, group in groups.items():
        for name in {canonical.get(key, {}).get("name", "")} | {row["artist"] for row in group}:
            if name:
                names[norm(name)].add(key)
        titles[key] = set().union(*(title_variants(row["title"]) for row in group))
    output = ROOT / "storage/public/merged/recording_verified_artist_locations.json"
    prior = json.loads(output.read_text())
    verified = defaultdict(dict)
    seen = set()

    def ingest(recordings):
        for recording in recordings:
            if recording["id"] in seen:
                continue
            seen.add(recording["id"])
            title = norm(recording.get("title"))
            for credit in recording.get("artist-credit", []):
                if not isinstance(credit, dict) or not credit.get("artist"):
                    continue
                artist = credit["artist"]
                keys = names.get(norm(artist.get("name")), set()) | names.get(norm(credit.get("name")), set())
                for key in keys:
                    if title in titles[key]:
                        verified[key][artist["id"]] = {
                            "recording_title": recording["title"],
                            "recording_url": "https://musicbrainz.org/recording/" + recording["id"],
                            "artist_url": "https://musicbrainz.org/artist/" + artist["id"],
                        }

    for path in CACHE.glob("recording-*.json"):
        ingest(json.loads(path.read_text()).get("recordings", []))
    already_verified = {
        key for key in groups if prior["audit"].get(key, {}).get("recording_evidence", {}).get("artist_url")
    }
    needed = set(groups) - set(verified) - already_verified

    # Recover unchanged original batch clauses from their content-addressed
    # filenames. Previously reviewed first-page groups may have been skipped,
    # so test every alignment rather than constructing a new set of requests.
    checks = []
    for key in sorted(old_groups, key=lambda k: -len(old_groups[k])):
        group = old_groups[key]
        search_names = (
            [canonical[key]["name"]] if key in canonical else list(dict.fromkeys(r["artist"] for r in group))[:2]
        )
        for name in search_names:
            for title in list(dict.fromkeys(r["title"] for r in group))[:2]:
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
    filenames = {p.name for p in CACHE.glob("recording-*.json")}
    repairs = {}
    for start in range(max(0, len(checks) - 14)):
        chunk = checks[start : start + 15]
        if not any(key in needed for key, _, _ in chunk):
            continue
        text = " OR ".join(
            "(artist:" + phrase(name) + " AND recording:" + phrase(title) + ")" for _, name, title in chunk
        )
        filename = "recording-" + hashlib.sha256(text.encode()).hexdigest() + ".json"
        if filename in filenames:
            response = json.loads((CACHE / filename).read_text())
            if int(response.get("count", 0)) > 100 and not response.get("split_complete"):
                repairs[filename] = [check for check in chunk if check[0] in needed]
    print(f"Pending artist groups: {len(groups)}; truncated cached batches to recover: {len(repairs)}", flush=True)
    for i, chunk in enumerate(repairs.values(), 1):
        remaining = [check for check in chunk if check[0] not in verified]
        if remaining:
            text = " OR ".join(
                "(artist:" + phrase(name) + " AND recording:" + phrase(title) + ")" for _, name, title in remaining
            )
            ingest(search("recording", text).get("recordings", []))
        if i % 5 == 0 or i == len(repairs):
            print(f"Truncated batch repair {i}/{len(repairs)}", flush=True)

    for path in CACHE.glob("recording-*.json"):
        ingest(json.loads(path.read_text()).get("recordings", []))
    # A geography-only pass may have enriched saved identities while this
    # read-only search was running. Preserve its latest records and audits.
    prior = json.loads(output.read_text())
    added = 0
    for key, matches in verified.items():
        if len(matches) != 1:
            continue
        if key in prior["artists"]:
            continue
        evidence = next(iter(matches.values()))
        old = prior["audit"].get(key, {}).get("recording_evidence", {})
        added += int(old.get("artist_url") != evidence["artist_url"])
        prior["audit"][key] = {
            "artist": groups[key][0]["artist"],
            "status": "recording_verified_pending_location",
            "search_name": canonical.get(key, {}).get("name") or groups[key][0]["artist"],
            "recording_evidence": evidence,
        }
    atomic(output, prior)
    report = {
        "pending_artist_groups": len(groups),
        "truncated_batches_repaired": len(repairs),
        "new_verified_identities": added,
        "unambiguous_cached_recording_matches": sum(len(v) == 1 for v in verified.values()),
    }
    atomic(ROOT / "tmp/artist-locations/truncation-repair-report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
