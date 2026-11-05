"""Inspect the catalog at track-ID grain and retain before/after genre evidence."""

from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcas_server.genre_metadata import apply_genre_metadata, genre_for_label, is_non_music

EAST_ASIAN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def audit(metadata: Path, output: Path) -> dict:
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    original_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
    clean = [apply_genre_metadata(dict(row)) for row in rows]
    labels = Counter(row["label"] for row in rows)
    ids = Counter(row["track_id"] for row in rows)
    recordings = Counter((row["artist"].strip().casefold(), row["title"].strip().casefold()) for row in rows)
    flags = []
    for raw, normal in zip(rows, clean):
        issues = []
        if EAST_ASIAN.search(raw.get("label_en", "")):
            issues.append("non_english_script_in_label_en")
        if not normal["duration_known"]:
            issues.append("unknown_duration")
        if is_non_music(raw):
            issues.append("spoken_drama_in_music_catalog")
        if not raw["release_date"]:
            issues.append("missing_release_date")
        if not normal["genre_normalized"]:
            issues.append("unresolved_genre")
        if issues:
            flags.append(
                {
                    "track_id": raw["track_id"],
                    "source_dataset": raw["source_dataset"],
                    "title": raw["title"],
                    "artist": raw["artist"],
                    "label_original": raw["label"],
                    "label_normalized": normal["label"],
                    "issues": ";".join(issues),
                }
            )
    summary = {
        "grain": "one platform track ID per row",
        "metadata_sha256": original_hash,
        "rows": len(rows),
        "sources": dict(Counter(row["source_dataset"] for row in rows)),
        "missing_required": {
            key: sum(not row[key].strip() for row in rows)
            for key in ["track_id", "title", "artist", "label", "culture"]
        },
        "duplicate_track_ids": sum(n - 1 for n in ids.values() if n > 1),
        "duplicate_complete_rows": len(rows) - len({tuple(row.values()) for row in rows}),
        "repeated_title_artist_groups": sum(n > 1 for n in recordings.values()),
        "repeated_title_artist_rows": sum(n for n in recordings.values() if n > 1),
        "raw_genre_variants": len(labels),
        "normalized_genres_used": len({row["genre_id"] for row in clean if row.get("genre_id")}),
        "changed_display_genre_rows": sum(a["label"] != b["label"] for a, b in zip(rows, clean)),
        "raw_label_en_nonempty": sum(bool(row["label_en"]) for row in rows),
        "raw_label_en_east_asian_script": sum(bool(EAST_ASIAN.search(row["label_en"])) for row in rows),
        "normalized_label_en_east_asian_script": sum(bool(EAST_ASIAN.search(row.get("label_en", ""))) for row in clean),
        "unresolved_genre_rows": sum(not row["genre_normalized"] for row in clean),
        "non_music_rows": sum(is_non_music(row) for row in rows),
        "unknown_duration_rows": sum(not row["duration_known"] for row in clean),
        "missing_release_date_rows": sum(not row["release_date"] for row in rows),
        "genre_kinds": dict(Counter(row.get("genre_kind", "unknown") for row in clean)),
        "rules": [
            "Normalize via public platform genre IDs plus reviewed aliases; do not infer from nationality.",
            "Keep raw inputs and original labels; filter explicit spoken drama only in music browsing/recommendation.",
            "Treat duration <= 0 as unknown; do not invent missing dates or merge recordings by title.",
        ],
        "limitations": [
            "No audio listening, fingerprint deduplication or factual audit of generated descriptions.",
            "Source category normalization does not independently reclassify musical content.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, values, fields in [
        (
            "flagged-tracks.csv",
            flags,
            ["track_id", "source_dataset", "title", "artist", "label_original", "label_normalized", "issues"],
        ),
        (
            "genre-variants.csv",
            [
                {
                    "original": label,
                    "tracks": count,
                    "canonical": genre_for_label(label)[1]["labels"]["en"],
                    "zh-CN": genre_for_label(label)[1]["labels"]["zh-CN"],
                    "genre_id": genre_for_label(label)[0],
                }
                for label, count in labels.most_common()
            ],
            ["original", "tracks", "canonical", "zh-CN", "genre_id"],
        ),
    ]:
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == original_hash
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "storage/public/merged/metadata_merged.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/catalog-quality")
    args = parser.parse_args()
    print(json.dumps(audit(args.metadata, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
