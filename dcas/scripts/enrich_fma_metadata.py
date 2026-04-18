from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FMA_METADATA_ZIP = REPO_ROOT / "tmp" / "fma_metadata.zip"

EXTRA_FIELDS = [
    "fma_track_id",
    "fma_album_id",
    "fma_artist_id",
    "fma_album_title",
    "fma_artist_location",
    "fma_artist_latitude",
    "fma_artist_longitude",
    "fma_artist_website",
    "fma_track_genre_top",
    "fma_track_listens",
    "fma_track_favorites",
    "fma_track_language_code",
    "fma_track_date_recorded",
    "fma_track_url",
    "fma_album_url",
    "fma_artist_url",
    "fma_match_method",
]


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _norm_text(value: Any) -> str:
    return " ".join(_clean(value).lower().split())


def _norm_url(value: Any) -> str:
    text = _clean(value)
    if text == "":
        return ""
    text = text.replace("https://", "http://")
    while text.endswith("/"):
        text = text[:-1]
    return text


def _load_fma_tracks(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        tracks = pd.read_csv(zf.open("fma_metadata/tracks.csv"), header=[0, 1], low_memory=False)
        raw_tracks = pd.read_csv(zf.open("fma_metadata/raw_tracks.csv"), low_memory=False)
    tracks = tracks.iloc[1:].copy()
    tracks.columns = [
        f"{a}__{b}" if not str(b).startswith("Unnamed") else str(a)
        for a, b in tracks.columns.to_list()
    ]
    tracks["track_id"] = pd.to_numeric(tracks["Unnamed: 0_level_0"], errors="coerce").astype("Int64")
    raw_subset = raw_tracks[["track_id", "track_url", "track_file", "album_url", "artist_url"]].copy()
    raw_subset["track_id"] = pd.to_numeric(raw_subset["track_id"], errors="coerce").astype("Int64")
    return tracks.merge(raw_subset, on="track_id", how="left")


def _candidate_score(row: pd.Series) -> tuple[float, float]:
    favorites = row.get("track__favorites")
    listens = row.get("track__listens")
    try:
        fav = float(favorites)
    except Exception:
        fav = 0.0
    try:
        lis = float(listens)
    except Exception:
        lis = 0.0
    return fav, lis


def _best_row(candidates: list[pd.Series]) -> pd.Series:
    if len(candidates) == 1:
        return candidates[0]
    ranked = sorted(candidates, key=_candidate_score, reverse=True)
    return ranked[0]


def _build_lookup(df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[tuple[str, str, str], pd.Series], dict[tuple[str, str], pd.Series]]:
    by_url: dict[str, pd.Series] = {}
    strict_groups: dict[tuple[str, str, str], list[pd.Series]] = {}
    loose_groups: dict[tuple[str, str], list[pd.Series]] = {}
    for _, row in df.iterrows():
        track_url = _norm_url(row.get("track_url"))
        if track_url:
            by_url[track_url] = row
        strict_key = (
            _norm_text(row.get("artist__name")),
            _norm_text(row.get("track__title")),
            _norm_text(row.get("album__title")),
        )
        loose_key = (
            _norm_text(row.get("artist__name")),
            _norm_text(row.get("track__title")),
        )
        strict_groups.setdefault(strict_key, []).append(row)
        loose_groups.setdefault(loose_key, []).append(row)
    by_strict = {key: _best_row(items) for key, items in strict_groups.items() if key != ("", "", "")}
    by_loose = {key: _best_row(items) for key, items in loose_groups.items() if key != ("", "")}
    return by_url, by_strict, by_loose


def _extract_fields(row: pd.Series, match_method: str) -> dict[str, str]:
    return {
        "fma_track_id": _clean(row.get("track_id")),
        "fma_album_id": _clean(row.get("album__id")),
        "fma_artist_id": _clean(row.get("artist__id")),
        "fma_album_title": _clean(row.get("album__title")),
        "fma_artist_location": _clean(row.get("artist__location")),
        "fma_artist_latitude": _clean(row.get("artist__latitude")),
        "fma_artist_longitude": _clean(row.get("artist__longitude")),
        "fma_artist_website": _clean(row.get("artist__website")),
        "fma_track_genre_top": _clean(row.get("track__genre_top")),
        "fma_track_listens": _clean(row.get("track__listens")),
        "fma_track_favorites": _clean(row.get("track__favorites")),
        "fma_track_language_code": _clean(row.get("track__language_code")),
        "fma_track_date_recorded": _clean(row.get("track__date_recorded")),
        "fma_track_url": _clean(row.get("track_url")),
        "fma_album_url": _clean(row.get("album_url")),
        "fma_artist_url": _clean(row.get("artist_url")),
        "fma_match_method": match_method,
    }


def _match_fma_row(
    item: dict[str, str],
    by_url: dict[str, pd.Series],
    by_strict: dict[tuple[str, str, str], pd.Series],
    by_loose: dict[tuple[str, str], pd.Series],
) -> tuple[pd.Series | None, str]:
    url_key = _norm_url(item.get("url"))
    if url_key and url_key in by_url:
        return by_url[url_key], "url"
    strict_key = (
        _norm_text(item.get("artist")),
        _norm_text(item.get("title")),
        _norm_text(item.get("fma_album_title") or item.get("album_title")),
    )
    if strict_key in by_strict:
        return by_strict[strict_key], "artist_title_album"
    loose_key = (
        _norm_text(item.get("artist")),
        _norm_text(item.get("title")),
    )
    if loose_key in by_loose:
        return by_loose[loose_key], "artist_title"
    return None, ""


def enrich_fma_metadata(
    metadata_csv: str | Path,
    out_csv: str | Path | None = None,
    fma_metadata_zip: str | Path = DEFAULT_FMA_METADATA_ZIP,
) -> dict[str, Any]:
    in_path = Path(metadata_csv)
    out_path = Path(out_csv) if out_csv is not None else in_path
    rows, fieldnames = _read_rows(in_path)

    fma_df = _load_fma_tracks(Path(fma_metadata_zip))
    by_url, by_strict, by_loose = _build_lookup(fma_df)

    enriched_rows: list[dict[str, str]] = []
    fma_rows = 0
    matched_rows = 0
    match_methods: Counter[str] = Counter()
    per_culture_artist_ids: dict[str, set[str]] = {}
    per_culture_locations: dict[str, Counter[str]] = {}
    field_nonempty: Counter[str] = Counter()
    overall_artist_ids: set[str] = set()
    overall_locations: Counter[str] = Counter()

    for row in rows:
        item = dict(row)
        is_fma = _norm_text(item.get("source_dataset")) == "free music archive"
        if is_fma:
            fma_rows += 1
            matched, method = _match_fma_row(item, by_url=by_url, by_strict=by_strict, by_loose=by_loose)
            if matched is not None:
                item.update(_extract_fields(matched, match_method=method))
                matched_rows += 1
                match_methods[method] += 1
                culture = _clean(item.get("culture"))
                artist_id = _clean(item.get("fma_artist_id"))
                location = _clean(item.get("fma_artist_location"))
                if culture not in per_culture_artist_ids:
                    per_culture_artist_ids[culture] = set()
                if culture not in per_culture_locations:
                    per_culture_locations[culture] = Counter()
                if artist_id:
                    per_culture_artist_ids[culture].add(artist_id)
                    overall_artist_ids.add(artist_id)
                if location:
                    per_culture_locations[culture][location] += 1
                    overall_locations[location] += 1
                for field in EXTRA_FIELDS:
                    if _clean(item.get(field)):
                        field_nonempty[field] += 1
            else:
                item["fma_match_method"] = ""
        enriched_rows.append(item)

    final_fields = list(fieldnames)
    for extra in EXTRA_FIELDS:
        if extra not in final_fields:
            final_fields.append(extra)

    write_path = out_path
    if out_path.resolve() == in_path.resolve():
        write_path = out_path.with_suffix(out_path.suffix + ".tmp")
    _write_rows(write_path, enriched_rows, final_fields)
    if write_path != out_path:
        os.replace(write_path, out_path)

    coverage = {
        field: (float(field_nonempty[field]) / float(fma_rows) if fma_rows else 0.0)
        for field in EXTRA_FIELDS
    }
    per_culture_report = []
    for culture in sorted(per_culture_artist_ids.keys() | per_culture_locations.keys()):
        artist_ids = per_culture_artist_ids.get(culture, set())
        locations = per_culture_locations.get(culture, Counter())
        culture_rows = sum(
            1
            for row in enriched_rows
            if _norm_text(row.get("source_dataset")) == "free music archive" and _clean(row.get("culture")) == culture
        )
        location_rows = sum(locations.values())
        geo_rows = sum(
            1
            for row in enriched_rows
            if _norm_text(row.get("source_dataset")) == "free music archive"
            and _clean(row.get("culture")) == culture
            and _clean(row.get("fma_artist_latitude"))
            and _clean(row.get("fma_artist_longitude"))
        )
        per_culture_report.append(
            {
                "culture": culture,
                "n_fma_rows": culture_rows,
                "unique_fma_artist_ids": len(artist_ids),
                "unique_artist_locations": len(locations),
                "artist_location_coverage": (float(location_rows) / float(culture_rows) if culture_rows else 0.0),
                "geo_coordinate_coverage": (float(geo_rows) / float(culture_rows) if culture_rows else 0.0),
                "top_artist_locations": [
                    {"location": loc, "count": int(count)}
                    for loc, count in locations.most_common(15)
                ],
            }
        )

    report = {
        "metadata_in": str(in_path.resolve()),
        "metadata_out": str(out_path.resolve()),
        "fma_metadata_zip": str(Path(fma_metadata_zip).resolve()),
        "n_rows": len(enriched_rows),
        "n_fma_rows": fma_rows,
        "n_matched_rows": matched_rows,
        "match_rate": (float(matched_rows) / float(fma_rows) if fma_rows else 0.0),
        "match_methods": dict(match_methods),
        "overall_unique_fma_artist_ids": len(overall_artist_ids),
        "overall_unique_artist_locations": len(overall_locations),
        "overall_top_artist_locations": [
            {"location": loc, "count": int(count)}
            for loc, count in overall_locations.most_common(25)
        ],
        "field_coverage_on_fma_rows": coverage,
        "per_culture": per_culture_report,
    }

    report_path = out_path.with_suffix(out_path.suffix + ".fma_enrichment_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# FMA Metadata Enrichment Report",
        "",
        f"- metadata: `{out_path}`",
        f"- FMA rows: `{fma_rows}`",
        f"- matched rows: `{matched_rows}`",
        f"- match rate: `{report['match_rate']:.2%}`",
        f"- unique FMA artist ids: `{report['overall_unique_fma_artist_ids']}`",
        f"- unique artist locations: `{report['overall_unique_artist_locations']}`",
        "",
        "## Field Coverage",
        "",
    ]
    for field in EXTRA_FIELDS:
        lines.append(f"- `{field}`: `{coverage[field]:.2%}`")
    lines.append("")
    lines.append("## Per-Culture Summary")
    lines.append("")
    for item in per_culture_report:
        lines.append(
            f"- `{item['culture']}`: rows=`{item['n_fma_rows']}`, "
            f"unique_artists=`{item['unique_fma_artist_ids']}`, "
            f"unique_locations=`{item['unique_artist_locations']}`, "
            f"location_coverage=`{item['artist_location_coverage']:.2%}`, "
            f"geo_coverage=`{item['geo_coordinate_coverage']:.2%}`"
        )
    md_path = out_path.with_suffix(out_path.suffix + ".fma_enrichment_report.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    report["report_json"] = str(report_path.resolve())
    report["report_md"] = str(md_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill FMA artist/location metadata into an existing metadata.csv file.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fma_metadata_zip", default=str(DEFAULT_FMA_METADATA_ZIP))
    args = ap.parse_args()

    report = enrich_fma_metadata(
        metadata_csv=args.metadata,
        out_csv=args.out,
        fma_metadata_zip=args.fma_metadata_zip,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
