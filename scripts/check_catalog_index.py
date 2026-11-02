"""Functional checks for filters, ordering, concurrent readers and metadata reload."""

from __future__ import annotations
import csv
import os
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcas_server.lightweight_catalog import LightweightMainlineCatalog
from dcas_server.paths import Storage


def main():
    with tempfile.TemporaryDirectory(prefix="echo-catalog-index-") as folder:
        root = Path(folder)
        path = root / "public/merged/metadata_merged.csv"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "track_id": "one",
                "title": "Quiet_Night",
                "artist": "Queen",
                "culture": "west",
                "source_dataset": "itunes",
                "description": "Calm music",
            },
            {
                "track_id": "two",
                "title": "Queen of the Sea",
                "artist": "Someone Else",
                "culture": "brazil",
                "source_dataset": "jamendo",
            },
            {
                "track_id": "three",
                "title": "Karaoke Version",
                "artist": "Covers",
                "culture": "west",
                "source_dataset": "itunes",
            },
        ]
        fields = [
            "track_id",
            "title",
            "artist",
            "culture",
            "source_dataset",
            "description",
        ]

        def write(items):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(items)

        write(rows)
        catalog = LightweightMainlineCatalog(Storage(root))
        assert [x["track_id"] for x in catalog.catalog(random_seed=None)["items"]] == [
            "one",
            "two",
        ]
        assert catalog.catalog(exclude_low_signal=False)["total_available"] == 3
        assert catalog.catalog(culture="west")["total_available"] == 1
        assert catalog.catalog(source_dataset="jamendo")["items"][0]["track_id"] == "two"
        assert catalog.catalog(culture="not-a-culture")["items"] == []
        assert catalog.catalog(q="quiet night")["items"][0]["track_id"] == "one"
        assert catalog.catalog(q="Queen")["items"][0]["artist"] == "Queen"
        expected = [x["track_id"] for x in catalog.catalog(random_seed=123)["items"]]
        with ThreadPoolExecutor(max_workers=20) as pool:
            actual = list(
                pool.map(
                    lambda _: [x["track_id"] for x in catalog.catalog(random_seed=123)["items"]],
                    range(40),
                )
            )
        assert all(x == expected for x in actual)
        write(
            [
                {
                    "track_id": "new",
                    "title": "New Sun",
                    "artist": "New Artist",
                    "culture": "china",
                    "source_dataset": "jamendo",
                }
            ]
        )
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        assert catalog.catalog(q="new sun")["items"][0]["track_id"] == "new"
        assert catalog.catalog(culture="west")["total_available"] == 0
        assert catalog.status()["n_tracks"] == 1
    print("PASS catalog filters, exact artist priority, deterministic concurrent readers and metadata invalidation")


if __name__ == "__main__":
    main()
