from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "track_id",
    "culture",
    "audio_path",
    "source_dataset",
    "source_split",
    "source_index",
    "duration_sec",
    "sample_rate",
    "channels",
    "coarse_label",
    "era",
    "region",
]

RECOMMENDED_FIELDS = [
    "fine_label",
    "label",
    "substyle",
    "instrument",
    "instrument_family",
    "language",
    "title",
    "artist",
    "license",
    "license_note",
    "url",
    "is_instrumental",
    "recording_condition",
]

GOVERNANCE_FIELDS = [
    "schema_version",
    "dataset_version",
    "import_batch",
    "dedup_group_id",
    "dedup_keep",
    "qc_status",
    "qc_notes",
    "embedding_status_culturemert",
    "embedding_status_gemini",
    "drop_reason",
]


@dataclass(frozen=True)
class MetadataAuditThresholds:
    min_tracks_per_culture: int = 30
    max_culture_imbalance_ratio: float = 20.0
    min_interactions_per_user: int = 5
    max_unknown_track_ratio: float = 0.01
    max_duplicate_user_track_ratio: float = 0.05


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _coverage(rows: list[dict[str, str]], field: str) -> float:
    if not rows:
        return 0.0
    filled = 0
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value != "" and value.lower() not in {"nan", "none", "null"}:
            filled += 1
    return round(float(filled / max(1, len(rows))), 6)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = sorted(values)
    def _at(q: float) -> float:
        idx = int(round((len(arr) - 1) * q))
        return round(float(arr[idx]), 6)
    return {
        "min": _at(0.0),
        "p25": _at(0.25),
        "p50": _at(0.5),
        "p75": _at(0.75),
        "max": _at(1.0),
        "mean": round(float(sum(arr) / len(arr)), 6),
    }


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _normalized_entropy(counter: Counter[str]) -> float:
    total = float(sum(counter.values()))
    if total <= 0 or len(counter) <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = float(count) / total
        entropy -= p * math.log(p + 1e-12)
    max_entropy = math.log(len(counter))
    if max_entropy <= 0:
        return 0.0
    return round(float(entropy / max_entropy), 6)


def _field_coverage_report(rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, Any]:
    fields_present = set(fieldnames)
    report = {
        "required": {},
        "recommended": {},
        "governance": {},
        "missing_required": sorted([field for field in REQUIRED_FIELDS if field not in fields_present]),
    }
    for group_name, fields in [
        ("required", REQUIRED_FIELDS),
        ("recommended", RECOMMENDED_FIELDS),
        ("governance", GOVERNANCE_FIELDS),
    ]:
        for field in fields:
            report[group_name][field] = {
                "present": field in fields_present,
                "coverage": _coverage(rows, field) if field in fields_present else 0.0,
            }
    return report


def _metadata_report(
    metadata_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    thresholds: MetadataAuditThresholds,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    n_rows = len(rows)

    track_counter = Counter(str(row.get("track_id", "")).strip() for row in rows if str(row.get("track_id", "")).strip())
    audio_counter = Counter(str(row.get("audio_path", "")).strip() for row in rows if str(row.get("audio_path", "")).strip())
    duplicate_track_ids = int(sum(1 for _, count in track_counter.items() if count > 1))
    duplicate_audio_paths = int(sum(1 for _, count in audio_counter.items() if count > 1))

    if duplicate_track_ids > 0:
        issues.append(
            {
                "severity": "error",
                "code": "metadata.duplicate_track_id",
                "message": f"duplicate track_id count = {duplicate_track_ids}",
            }
        )

    culture_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    culture_source_counter: dict[str, Counter[str]] = defaultdict(Counter)
    duration_values: list[float] = []
    sample_rate_counter: Counter[str] = Counter()
    channel_counter: Counter[str] = Counter()
    missing_audio = 0

    for row in rows:
        culture = str(row.get("culture", "")).strip()
        source = str(row.get("source_dataset", "")).strip()
        if culture:
            culture_counter[culture] += 1
        if source:
            source_counter[source] += 1
        if culture and source:
            culture_source_counter[culture][source] += 1

        duration = _safe_float(row.get("duration_sec"))
        if duration is not None:
            duration_values.append(duration)
        sample_rate = str(row.get("sample_rate", "")).strip()
        if sample_rate:
            sample_rate_counter[sample_rate] += 1
        channels = str(row.get("channels", "")).strip()
        if channels:
            channel_counter[channels] += 1

        audio_path = str(row.get("audio_path", "")).strip()
        if audio_path == "":
            missing_audio += 1
            continue
        resolved = Path(audio_path)
        if not resolved.is_absolute():
            resolved = (metadata_path.parent / resolved).resolve()
        if not resolved.exists():
            missing_audio += 1

    culture_distribution = [
        {"culture": culture, "count": int(count), "ratio": round(float(count / max(1, n_rows)), 6)}
        for culture, count in sorted(culture_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    source_distribution = [
        {"source_dataset": source, "count": int(count), "ratio": round(float(count / max(1, n_rows)), 6)}
        for source, count in sorted(source_counter.items(), key=lambda item: (-item[1], item[0]))
    ]

    if culture_counter:
        culture_imbalance_ratio = round(float(max(culture_counter.values()) / max(1, min(culture_counter.values()))), 6)
    else:
        culture_imbalance_ratio = 0.0

    if culture_imbalance_ratio > thresholds.max_culture_imbalance_ratio:
        issues.append(
            {
                "severity": "warn",
                "code": "metadata.culture_imbalance",
                "message": f"culture imbalance ratio is high ({culture_imbalance_ratio})",
            }
        )

    for culture, count in sorted(culture_counter.items()):
        if count < thresholds.min_tracks_per_culture:
            issues.append(
                {
                    "severity": "warn",
                    "code": "metadata.culture_low_count",
                    "message": f"culture '{culture}' has only {count} tracks",
                }
            )

    if missing_audio > 0:
        issues.append(
            {
                "severity": "warn",
                "code": "metadata.audio_missing",
                "message": f"rows with missing audio files = {missing_audio}",
            }
        )

    top_source_share_by_culture: list[dict[str, Any]] = []
    for culture, sources in sorted(culture_source_counter.items()):
        total = max(1, sum(sources.values()))
        top_source, top_count = sorted(sources.items(), key=lambda item: (-item[1], item[0]))[0]
        top_source_share_by_culture.append(
            {
                "culture": culture,
                "top_source_dataset": top_source,
                "top_source_share": round(float(top_count / total), 6),
                "n_sources": int(len(sources)),
                "source_entropy_norm": _normalized_entropy(sources),
            }
        )

    top_culture_share_by_source: list[dict[str, Any]] = []
    source_culture_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for culture, sources in culture_source_counter.items():
        for source, count in sources.items():
            source_culture_counter[source][culture] += count
    for source, cultures in sorted(source_culture_counter.items()):
        total = max(1, sum(cultures.values()))
        top_culture, top_count = sorted(cultures.items(), key=lambda item: (-item[1], item[0]))[0]
        top_culture_share_by_source.append(
            {
                "source_dataset": source,
                "top_culture": top_culture,
                "top_culture_share": round(float(top_count / total), 6),
                "n_cultures": int(len(cultures)),
            }
        )

    source_confound_report = {
        "culture_source_matrix": {
            culture: dict(sorted(counter.items()))
            for culture, counter in sorted(culture_source_counter.items())
        },
        "single_source_culture_count": int(sum(1 for counter in culture_source_counter.values() if len(counter) <= 1)),
        "top_source_share_by_culture": top_source_share_by_culture,
        "top_culture_share_by_source": top_culture_share_by_source,
        "weighted_source_predictability_from_culture": round(
            float(
                sum(item["top_source_share"] * culture_counter[item["culture"]] for item in top_source_share_by_culture)
                / max(1, sum(culture_counter.values()))
            ),
            6,
        ) if top_source_share_by_culture else 0.0,
        "weighted_culture_predictability_from_source": round(
            float(
                sum(item["top_culture_share"] * source_counter[item["source_dataset"]] for item in top_culture_share_by_source)
                / max(1, sum(source_counter.values()))
            ),
            6,
        ) if top_culture_share_by_source else 0.0,
    }
    if int(source_confound_report["single_source_culture_count"]) > 0:
        issues.append(
            {
                "severity": "warn",
                "code": "metadata.single_source_culture",
                "message": f"{int(source_confound_report['single_source_culture_count'])} cultures are backed by a single source dataset",
            }
        )
    if float(source_confound_report["weighted_source_predictability_from_culture"]) >= 0.8:
        issues.append(
            {
                "severity": "warn",
                "code": "metadata.source_confound_high",
                "message": (
                    "culture-to-source predictability is high "
                    f"({float(source_confound_report['weighted_source_predictability_from_culture'])})"
                ),
            }
        )

    field_coverage = _field_coverage_report(rows, fieldnames)
    for field in field_coverage["missing_required"]:
        issues.append(
            {
                "severity": "error",
                "code": "metadata.required_field_missing",
                "message": f"required field is missing: {field}",
            }
        )
    for field, stats in field_coverage["required"].items():
        if not stats["present"]:
            continue
        coverage = float(stats["coverage"])
        if coverage < 1.0:
            issues.append(
                {
                    "severity": "warn",
                    "code": "metadata.required_field_incomplete",
                    "message": f"required field '{field}' coverage is only {coverage}",
                }
            )

    profile = {
        "metadata_path": _safe_rel(metadata_path),
        "n_rows": int(n_rows),
        "n_cultures": int(len(culture_counter)),
        "n_sources": int(len(source_counter)),
        "culture_distribution": culture_distribution,
        "source_distribution": source_distribution,
        "duration_sec_stats": _quantiles(duration_values),
        "sample_rate_distribution": dict(sorted(sample_rate_counter.items())),
        "channel_distribution": dict(sorted(channel_counter.items())),
        "culture_imbalance_ratio": culture_imbalance_ratio,
        "missing_audio_rows": int(missing_audio),
    }

    duplicates = {
        "duplicate_track_id_count": int(duplicate_track_ids),
        "duplicate_audio_path_count": int(duplicate_audio_paths),
    }

    missingness = field_coverage
    schema = {
        "fieldnames": fieldnames,
        "required_fields": REQUIRED_FIELDS,
        "recommended_fields": RECOMMENDED_FIELDS,
        "governance_fields": GOVERNANCE_FIELDS,
    }
    return profile, schema, missingness, duplicates, source_confound_report, issues


def _interactions_report(
    path: Path,
    known_tracks: set[str],
    track_to_culture: dict[str, str],
    thresholds: MetadataAuditThresholds,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    rows, fieldnames = _read_csv(path)
    if not rows:
        issues.append(
            {
                "severity": "warn",
                "code": "interactions.empty",
                "message": f"{path.name} is empty",
            }
        )
    required = {"user_id", "track_id"}
    missing = sorted(required - set(fieldnames))
    if missing:
        issues.append(
            {
                "severity": "error",
                "code": "interactions.required_field_missing",
                "message": f"{path.name} missing required fields: {missing}",
            }
        )
        return {"path": _safe_rel(path), "n_rows": 0, "n_users": 0}, issues

    unknown_track_count = 0
    user_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()
    seen_tracks: set[str] = set()
    culture_counter: Counter[str] = Counter()

    for row in rows:
        user_id = str(row.get("user_id", "")).strip()
        track_id = str(row.get("track_id", "")).strip()
        if user_id:
            user_counter[user_id] += 1
        if user_id and track_id:
            pair_counter[(user_id, track_id)] += 1
        if track_id:
            seen_tracks.add(track_id)
        if track_id not in known_tracks:
            unknown_track_count += 1
        culture = track_to_culture.get(track_id, "")
        if culture:
            culture_counter[culture] += 1

    duplicate_pairs = int(sum(1 for _, count in pair_counter.items() if count > 1))
    duplicate_ratio = round(float(duplicate_pairs / max(1, len(rows))), 6)
    unknown_track_ratio = round(float(unknown_track_count / max(1, len(rows))), 6)
    per_user_values = list(user_counter.values())
    per_user_stats = _quantiles([float(value) for value in per_user_values])
    if user_counter and min(per_user_values) < thresholds.min_interactions_per_user:
        issues.append(
            {
                "severity": "warn",
                "code": "interactions.user_low_activity",
                "message": f"{path.name} has users with fewer than {thresholds.min_interactions_per_user} interactions",
            }
        )
    if unknown_track_ratio > thresholds.max_unknown_track_ratio:
        issues.append(
            {
                "severity": "warn",
                "code": "interactions.unknown_track_ratio_high",
                "message": f"{path.name} unknown_track_ratio = {unknown_track_ratio}",
            }
        )
    if duplicate_ratio > thresholds.max_duplicate_user_track_ratio:
        issues.append(
            {
                "severity": "warn",
                "code": "interactions.duplicate_pair_ratio_high",
                "message": f"{path.name} duplicate_user_track_ratio = {duplicate_ratio}",
            }
        )

    report = {
        "path": _safe_rel(path),
        "n_rows": int(len(rows)),
        "n_users": int(len(user_counter)),
        "n_tracks_observed": int(len(seen_tracks)),
        "track_coverage_ratio": round(float(len(seen_tracks.intersection(known_tracks)) / max(1, len(known_tracks))), 6),
        "unknown_track_ratio": unknown_track_ratio,
        "duplicate_user_track_ratio": duplicate_ratio,
        "per_user_interaction_stats": per_user_stats,
        "culture_exposure_distribution": [
            {"culture": culture, "count": int(count), "ratio": round(float(count / max(1, len(rows))), 6)}
            for culture, count in sorted(culture_counter.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
    return report, issues


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.extend(["# Dataset Profile", ""])
    lines.append(f"- dataset_name: `{report.get('dataset_name', '')}`")
    lines.append(f"- metadata_rows: `{int(report['profile'].get('n_rows', 0))}`")
    lines.append(f"- cultures: `{int(report['profile'].get('n_cultures', 0))}`")
    lines.append(f"- sources: `{int(report['profile'].get('n_sources', 0))}`")
    lines.append("")

    lines.extend(["## Culture Distribution", "", "| culture | count | ratio |", "|---|---:|---:|"])
    for row in report["profile"].get("culture_distribution", []):
        lines.append(f"| {row['culture']} | {row['count']} | {row['ratio']} |")

    lines.extend(["", "## Source Distribution", "", "| source_dataset | count | ratio |", "|---|---:|---:|"])
    for row in report["profile"].get("source_distribution", []):
        lines.append(f"| {row['source_dataset']} | {row['count']} | {row['ratio']} |")

    confound = report.get("source_confound", {})
    lines.extend(["", "## Source Confound", ""])
    lines.append(f"- single_source_culture_count: `{int(confound.get('single_source_culture_count', 0))}`")
    lines.append(f"- weighted_source_predictability_from_culture: `{confound.get('weighted_source_predictability_from_culture', 0.0)}`")
    lines.append(f"- weighted_culture_predictability_from_source: `{confound.get('weighted_culture_predictability_from_source', 0.0)}`")
    lines.append("")
    lines.append("| culture | top_source_dataset | top_source_share | n_sources | source_entropy_norm |")
    lines.append("|---|---|---:|---:|---:|")
    for row in confound.get("top_source_share_by_culture", []):
        lines.append(
            f"| {row['culture']} | {row['top_source_dataset']} | {row['top_source_share']} | {row['n_sources']} | {row['source_entropy_norm']} |"
        )

    interactions = report.get("interactions", [])
    if interactions:
        lines.extend(["", "## Interactions", ""])
        for item in interactions:
            lines.append(f"### `{Path(item['path']).name}`")
            lines.append("")
            lines.append(f"- rows: `{item['n_rows']}`")
            lines.append(f"- users: `{item['n_users']}`")
            lines.append(f"- track_coverage_ratio: `{item['track_coverage_ratio']}`")
            lines.append(f"- unknown_track_ratio: `{item['unknown_track_ratio']}`")
            lines.append(f"- duplicate_user_track_ratio: `{item['duplicate_user_track_ratio']}`")
            lines.append("")

    lines.extend(["## Issues", ""])
    issues = report.get("issues", [])
    if not issues:
        lines.append("- none")
    else:
        lines.append("| severity | code | message |")
        lines.append("|---|---|---|")
        for issue in issues:
            lines.append(f"| {issue['severity']} | {issue['code']} | {issue['message']} |")
    lines.append("")
    return "\n".join(lines)


def audit_dataset_v4(
    metadata_csv: str | Path,
    out_dir: str | Path,
    interactions: list[str | Path] | None = None,
    dataset_name: str | None = None,
    thresholds: MetadataAuditThresholds | None = None,
) -> dict[str, Any]:
    th = thresholds or MetadataAuditThresholds()
    metadata_path = Path(metadata_csv)
    rows, fieldnames = _read_csv(metadata_path)
    profile, schema, missingness, duplicates, source_confound, issues = _metadata_report(
        metadata_path=metadata_path,
        rows=rows,
        fieldnames=fieldnames,
        thresholds=th,
    )

    track_to_culture = {str(row.get("track_id", "")).strip(): str(row.get("culture", "")).strip() for row in rows}
    known_tracks = set(track_to_culture.keys())

    interaction_reports: list[dict[str, Any]] = []
    for item in interactions or []:
        path = Path(item)
        if not path.exists():
            issues.append(
                {
                    "severity": "warn",
                    "code": "interactions.not_found",
                    "message": f"interaction file not found: {path}",
                }
            )
            continue
        report, inter_issues = _interactions_report(
            path=path,
            known_tracks=known_tracks,
            track_to_culture=track_to_culture,
            thresholds=th,
        )
        interaction_reports.append(report)
        issues.extend(inter_issues)

    report = {
        "dataset_name": dataset_name or metadata_path.parent.name,
        "profile": profile,
        "schema": schema,
        "missingness": missingness,
        "duplicates": duplicates,
        "source_confound": source_confound,
        "interactions": interaction_reports,
        "issues": issues,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "dataset_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "dataset_profile.md").write_text(_to_markdown(report), encoding="utf-8")
    (out_path / "schema_report.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "missingness_report.json").write_text(json.dumps(missingness, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "duplicate_report.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "source_confound_report.json").write_text(json.dumps(source_confound, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit V4-style metadata and interaction files.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--interactions", nargs="*", default=None)
    ap.add_argument("--min_tracks_per_culture", type=int, default=30)
    ap.add_argument("--max_culture_imbalance_ratio", type=float, default=20.0)
    ap.add_argument("--min_interactions_per_user", type=int, default=5)
    ap.add_argument("--max_unknown_track_ratio", type=float, default=0.01)
    ap.add_argument("--max_duplicate_user_track_ratio", type=float, default=0.05)
    args = ap.parse_args()

    thresholds = MetadataAuditThresholds(
        min_tracks_per_culture=int(args.min_tracks_per_culture),
        max_culture_imbalance_ratio=float(args.max_culture_imbalance_ratio),
        min_interactions_per_user=int(args.min_interactions_per_user),
        max_unknown_track_ratio=float(args.max_unknown_track_ratio),
        max_duplicate_user_track_ratio=float(args.max_duplicate_user_track_ratio),
    )
    report = audit_dataset_v4(
        metadata_csv=args.metadata,
        out_dir=args.out_dir,
        interactions=list(args.interactions or []),
        dataset_name=args.dataset_name,
        thresholds=thresholds,
    )
    print(json.dumps({"out_dir": str(Path(args.out_dir).resolve()), "issues": len(report["issues"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
