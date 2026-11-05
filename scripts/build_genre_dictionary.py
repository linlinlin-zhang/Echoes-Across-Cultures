"""Build shared display metadata from public Apple genre IDs and reviewed catalog aliases.

The input taxonomies are downloaded once with --fetch to a local cache; building is offline.
Raw catalog CSVs and model inputs are never modified.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"en": "us", "zh-CN": "cn", "de": "de", "es": "es", "pt": "br", "ja": "jp", "ko": "kr"}
COUNTRIES = ["us", "cn", "tw", "de", "es", "br", "jp", "kr", "fr", "it", "se", "nl", "tr", "dk"]


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).lower().replace("’", "'").split())


def flatten(tree: dict) -> dict[str, str]:
    result = {}
    for key, row in tree.items():
        result[key] = row["name"]
        result.update(flatten(row.get("subgenres", {})))
    return result


def build(cache: Path, metadata: Path, output: Path) -> dict:
    reviewed_path = ROOT / "configs/genre_aliases.reviewed.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    taxonomies = {cc: flatten(json.loads((cache / (cc + ".json")).read_text(encoding="utf-8"))) for cc in COUNTRIES}
    english = dict(taxonomies["us"])
    english.update(reviewed["english_overrides"])
    groups = defaultdict(list)
    for gid, name in english.items():
        groups[normalize(name)].append(gid)
    genres, source_ids = {}, {}
    for key, ids in groups.items():
        chosen = min(ids, key=int)
        canonical_id = "apple:" + chosen
        labels = {lang: taxonomies[country].get(chosen, english[chosen]) for lang, country in LANGUAGES.items()}
        labels["en"] = english[chosen]
        labels.update(reviewed["genre_label_overrides"].get(chosen, {}))
        genres[canonical_id] = {
            "labels": labels,
            "kind": "spoken_drama" if canonical_id in reviewed["non_music_ids"] else "genre",
            "exclude_from_music": canonical_id in reviewed["non_music_ids"],
            "platform_ids": sorted(ids, key=int),
        }
        for gid in ids:
            source_ids[gid] = canonical_id
    english_ids = {normalize(g["labels"]["en"]): gid for gid, g in genres.items()}
    candidates = defaultdict(set)
    for taxonomy in taxonomies.values():
        for gid, label in taxonomy.items():
            candidates[normalize(label)].add(source_ids[gid])
    for genre in reviewed["custom_genres"]:
        gid = genre["id"]
        genres[gid] = {
            "labels": genre["labels"],
            "kind": genre["kind"],
            "exclude_from_music": False,
            "platform_ids": [],
        }
        english_ids[normalize(genre["labels"]["en"])] = gid
        for alias in genre["aliases"]:
            candidates[normalize(alias)].add(gid)
    aliases, ambiguous = {}, {}
    preferred = {normalize(k): v for k, v in reviewed["preferred_aliases"].items()}
    for alias, ids in candidates.items():
        if alias in preferred:
            aliases[alias] = english_ids[normalize(preferred[alias])]
        elif alias in english_ids:
            aliases[alias] = english_ids[alias]
        elif len(ids) == 1:
            aliases[alias] = next(iter(ids))
        elif "apple:7" in ids:
            aliases[alias] = "apple:7"  # Localized broad Electronic, not a guessed subgenre.
        else:
            ambiguous[alias] = sorted(ids)
    for alias, name in reviewed["extra_aliases"].items():
        aliases[normalize(alias)] = english_ids[normalize(name)]
    for gid, genre in genres.items():
        aliases.setdefault(normalize(genre["labels"]["en"]), gid)
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    unresolved = sorted({r["label"] for r in rows if normalize(r["label"]) not in aliases})
    if unresolved:
        raise ValueError("Unresolved catalog labels: " + json.dumps(unresolved, ensure_ascii=False))
    # Every displayed English label must be English/Latin-script, including reviewed proper names.
    for genre in genres.values():
        assert all(lang in genre["labels"] for lang in LANGUAGES)
        assert not any(
            "\u3400" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" or "\uac00" <= c <= "\ud7af"
            for c in genre["labels"]["en"]
        )
    data = {
        "schema_version": 1,
        "sources": [
            {
                "url": "https://itunes.apple.com/WebObjects/MZStoreServices.woa/ws/genres?id=34&cc=" + cc,
                "sha256": hashlib.sha256((cache / (cc + ".json")).read_bytes()).hexdigest(),
            }
            for cc in COUNTRIES
        ],
        "reviewed_aliases_sha256": hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
        "genres": dict(sorted(genres.items())),
        "aliases": dict(sorted(aliases.items())),
        "ambiguous_aliases": dict(sorted(ambiguous.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    output.with_suffix(".js").write_text(
        "window.EchoGenreData = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    return {
        "catalog_rows": len(rows),
        "raw_labels": len({r["label"] for r in rows}),
        "canonical_genres": len(genres),
        "aliases": len(aliases),
        "unresolved_catalog_labels": len(unresolved),
        "non_music_rows": sum(genres[aliases[normalize(r["label"])]]["exclude_from_music"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "storage/public/merged/metadata_merged.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "web/genre-data.json")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        for cc in COUNTRIES:
            path = args.cache_dir / (cc + ".json")
            if not path.exists():
                with urllib.request.urlopen(
                    "https://itunes.apple.com/WebObjects/MZStoreServices.woa/ws/genres?id=34&cc=" + cc, timeout=30
                ) as response:
                    path.write_bytes(response.read())
    print(json.dumps(build(args.cache_dir, args.metadata, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
