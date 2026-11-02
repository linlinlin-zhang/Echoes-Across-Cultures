"""Fill source-backed city labels without changing identities or coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from enrich_artist_locations import ROOT, atomic, batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "storage/public/merged/recording_verified_artist_locations.json"
    )
    parser.add_argument(
        "--missing-only", action="store_true", help="Reuse the saved city label index and fetch only missing places"
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    index_path = ROOT / "storage/artist_location_cache/city-labels.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    places = {}
    for record in data["artists"].values():
        match = re.search(r"/(?:wiki|entity)/(Q\d+)$", record.get("location_place_url", ""))
        if record.get("location_precision") == "artist_city" and match:
            if not args.missing_only or match.group(1) not in index:
                places[match.group(1)] = None
    languages = "mul|en|zh-hans|zh|de|es|pt|ja|ko"
    chunks = batches(sorted(places), 50)
    for i, chunk in enumerate(chunks, 1):
        params = dict(action="wbgetentities", ids="|".join(chunk), props="labels", languages=languages, format="json")
        suffix = urllib.parse.urlencode(params)
        cache = (
            ROOT
            / "storage/artist_location_cache"
            / ("city-labels-" + hashlib.sha256(suffix.encode()).hexdigest() + ".json")
        )
        if cache.exists():
            response = json.loads(cache.read_text())
        else:
            for attempt in range(3):
                try:
                    request = urllib.request.Request(
                        "https://www.wikidata.org/w/api.php?" + suffix,
                        headers={"User-Agent": "EchoMusicCatalog/1.0 (public city labels)"},
                    )
                    with urllib.request.urlopen(request, timeout=35) as stream:
                        response = json.load(stream)
                    if "entities" not in response:
                        raise ValueError("Missing city label results")
                    atomic(cache, response)
                    time.sleep(1.1)
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
        for qid, entity in response["entities"].items():
            labels = {lang: item["value"] for lang, item in entity.get("labels", {}).items()}
            shared = labels.pop("mul", None)
            if shared:
                for lang in ["en", "de", "es", "pt"]:
                    labels.setdefault(lang, shared)
            chinese = labels.pop("zh-hans", None) or labels.pop("zh", None)
            labels.pop("zh", None)
            if chinese:
                labels["zh-CN"] = chinese
            places[qid] = labels
        print(f"Public city labels {i}/{len(chunks)}", flush=True)
    count = 0
    for record in data["artists"].values():
        match = re.search(r"/(?:wiki|entity)/(Q\d+)$", record.get("location_place_url", ""))
        labels = (places.get(match.group(1)) or index.get(match.group(1))) if match else None
        if labels:
            record["city_labels"] = {**record.get("city_labels", {}), **labels}
            if not record.get("city") and labels.get("en"):
                record["city"] = labels["en"]
            count += 1
    index.update({qid: labels for qid, labels in places.items() if labels})
    atomic(index_path, index)
    atomic(args.input, data)
    print(json.dumps({"city_records_with_public_labels": count, "unique_places": len(places)}), flush=True)


if __name__ == "__main__":
    main()
