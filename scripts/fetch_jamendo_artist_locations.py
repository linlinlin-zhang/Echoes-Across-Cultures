"""Fetch Jamendo artists' self-declared geography by track/artist IDs (never fuzzy names)."""

from __future__ import annotations
import argparse
import csv
import http.client
import json
import os
import re
import time
import urllib.request
import urllib.parse
from enrich_artist_locations import ROOT, atomic, batches


def api(method, params, key, cache):
    # Never put credentials into logs or cache filenames.
    import hashlib

    path = cache / (
        "jamendo-"
        + method.replace("/", "-")
        + "-"
        + hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        + ".json"
    )
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            url = (
                "https://api.jamendo.com/v3.0/"
                + method
                + "/?"
                + urllib.parse.urlencode([("client_id", key), ("format", "json"), ("limit", "200")] + params)
            )
            with urllib.request.urlopen(url, timeout=40) as response:
                data = json.load(response)
            if data.get("headers", {}).get("status") != "success":
                raise ValueError("Jamendo API did not return success")
            results = data["results"]
            atomic(path, results)
            time.sleep(0.6)
            return results
        except (OSError, http.client.HTTPException, ValueError, KeyError):
            if attempt == 2:
                raise RuntimeError("Jamendo request failed; credentials were not logged") from None
            time.sleep(3 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help="Preserve saved profiles and retry unresolved track IDs in small batches",
    )
    args = parser.parse_args()
    key = os.getenv("JAMENDO_CLIENT_ID", "")
    if not key:
        text = (ROOT / "run_jamendo_crawl.ps1").read_text()
        match = re.search(r'\$JAMENDO_CLIENT_ID\s*=\s*"([^\"]+)"', text)
        key = match.group(1) if match else ""
    if not key:
        raise SystemExit("Existing JAMENDO_CLIENT_ID configuration is missing")
    cache = ROOT / "storage/artist_location_cache"
    rows = [
        r
        for r in csv.DictReader(
            (ROOT / "storage/public/merged/metadata_merged.csv").open(encoding="utf-8-sig", newline="")
        )
        if r["source_dataset"] == "jamendo"
    ]
    prior_path = ROOT / "storage/artist_location_cache/jamendo_artist_records.json"
    prior = (
        json.loads(prior_path.read_text())
        if args.repair_missing and prior_path.exists()
        else {"tracks": {}, "artists": {}}
    )
    mapping = dict(prior["tracks"])
    locations = dict(prior["artists"])
    artists = {aid: record["name"] for aid, record in locations.items()}
    ids = sorted({r["jamendo_id"] for r in rows if r.get("jamendo_id") and r["track_id"] not in mapping})

    def tracks(chunk):
        found = api("tracks", [("id[]", tid) for tid in chunk], key, cache)
        if not found and len(chunk) > 1:
            mid = len(chunk) // 2
            return tracks(chunk[:mid]) + tracks(chunk[mid:])
        return found

    for i, chunk in enumerate(batches(ids, 50), 1):
        for item in tracks(chunk):
            if str(item.get("id")) in chunk:
                mapping["jamendo_" + str(item["id"])] = "jamendo:" + str(item["artist_id"])
                artists[str(item["artist_id"])] = item["artist_name"]
        print(f"Jamendo artist IDs: batch {i}/{len(batches(ids, 50))}, tracks {len(mapping)}", flush=True)
    missing_artists = sorted(set(artists) - set(locations))
    for i, chunk in enumerate(batches(missing_artists, 50), 1):
        for item in api("artists/locations", [("id[]", aid) for aid in chunk], key, cache):
            if str(item["id"]) in chunk:
                locations[str(item["id"])] = item
        print(f"Jamendo public locations: batch {i}, artists {len(locations)}", flush=True)
    atomic(
        ROOT / "storage/artist_location_cache/jamendo_artist_records.json", {"tracks": mapping, "artists": locations}
    )
    print("Public Jamendo geography downloaded", len(mapping), "tracks", len(locations), "artist profiles", flush=True)


if __name__ == "__main__":
    main()
