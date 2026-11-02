"""Recover canonical artist names from Apple IDs; do not treat the store country as origin."""

import csv
import json
import hashlib
import time
import urllib.parse
import urllib.request
from enrich_artist_locations import ROOT, atomic, batches, artist_key


def main():
    rows = list(
        csv.DictReader((ROOT / "storage/public/merged/metadata_merged.csv").open(encoding="utf-8-sig", newline=""))
    )
    base = json.loads((ROOT / "storage/public/merged/artist_locations.json").read_text())
    ids = sorted({r["artist_id"] for r in rows if r.get("artist_id") and artist_key(r) not in base["artists"]})
    output = ROOT / "storage/artist_location_cache/canonical_artist_names.json"
    result = json.loads(output.read_text()) if output.exists() else {}
    for i, chunk in enumerate(batches(ids, 150), 1):
        cache = (
            ROOT
            / "storage/artist_location_cache"
            / ("apple-names-" + hashlib.sha256(",".join(chunk).encode()).hexdigest() + ".json")
        )
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            url = "https://itunes.apple.com/lookup?" + urllib.parse.urlencode(
                {"id": ",".join(chunk), "entity": "musicArtist", "country": "US", "lang": "en_us", "limit": 200}
            )
            with urllib.request.urlopen(url, timeout=35) as response:
                data = json.load(response)
            atomic(cache, data)
            time.sleep(3.2)
        for record in data.get("results", []):
            if record.get("wrapperType") == "artist" and str(record.get("artistId")) in chunk:
                result["itunes:" + str(record["artistId"])] = {
                    "name": record["artistName"],
                    "source_url": record.get("artistLinkUrl", ""),
                }
        atomic(output, result)
        print("Canonical Apple names:", i, "/", len(batches(ids, 150)), "records:", len(result), flush=True)


if __name__ == "__main__":
    main()
