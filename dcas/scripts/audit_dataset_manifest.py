from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SOURCE_FIELDS = {"track_id", "culture", "audio_path"}
OPTIONAL_AUDIT_FIELDS = [
    "source_dataset",
    "duration_sec",
    "coarse_label",
    "label",
    "title",
    "artist",
    "license",
    "language",
    "instrument",
    "instrument_family",
]
REQUIRED_EMBEDDING_FIELDS = {
    "culturemert": {
        "model_id",
        "pooling",
        "max_seconds",
        "window_count",
        "window_strategy",
        "window_aggregate",
    },
    "gemini": {
        "model_id",
        "output_dimensionality",
        "max_seconds",
        "window_count",
        "window_strategy",
        "window_aggregate",
    },
}
RECOMMENDED_TOP_LEVEL = {
    "merge",
    "harmonize",
    "interaction_protocol",
    "embeddings",
    "validation",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def _coverage(rows: list[dict[str, str]], field: str) -> float:
    if not rows:
        return 0.0
    filled = 0
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value != "" and value.lower() != "nan":
            filled += 1
    return float(filled / max(1, len(rows)))


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _audit_source(
    source: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    local_metadata = Path(str(source.get("local_metadata", "")))
    culture = str(source.get("culture", "")).strip()
    dataset_id = str(source.get("dataset_id", "")).strip()
    if not local_metadata.is_absolute():
        local_metadata = (REPO_ROOT / local_metadata).resolve()

    if not local_metadata.exists():
        issues.append(
            {
                "severity": "error",
                "code": "source.metadata_missing",
                "message": f"{dataset_id}: metadata file not found: {local_metadata}",
            }
        )
        return {
            "dataset_id": dataset_id,
            "culture": culture,
            "local_metadata": str(local_metadata),
            "exists": False,
        }, issues

    rows, fieldnames = _read_rows(local_metadata)
    fields = set(fieldnames)
    missing = sorted(REQUIRED_SOURCE_FIELDS - fields)
    if missing:
        issues.append(
            {
                "severity": "error",
                "code": "source.required_columns_missing",
                "message": f"{dataset_id}: missing required columns {missing}",
            }
        )

    track_counter = Counter(str(r.get("track_id", "")).strip() for r in rows if str(r.get("track_id", "")).strip())
    duplicate_track_ids = int(sum(1 for _, cnt in track_counter.items() if cnt > 1))
    if duplicate_track_ids > 0:
        issues.append(
            {
                "severity": "warn",
                "code": "source.duplicate_track_id",
                "message": f"{dataset_id}: duplicate track_id count = {duplicate_track_ids}",
            }
        )

    culture_values = Counter(str(r.get("culture", "")).strip() for r in rows if str(r.get("culture", "")).strip())
    if culture != "" and culture_values and set(culture_values.keys()) != {culture}:
        issues.append(
            {
                "severity": "warn",
                "code": "source.culture_mismatch",
                "message": f"{dataset_id}: metadata cultures {sorted(culture_values.keys())} do not match manifest culture '{culture}'",
            }
        )

    missing_audio = 0
    for row in rows:
        rel = str(row.get("audio_path", "")).strip()
        if rel == "":
            missing_audio += 1
            continue
        audio_path = Path(rel)
        if not audio_path.is_absolute():
            audio_path = (local_metadata.parent / audio_path).resolve()
        if not audio_path.exists():
            missing_audio += 1
    if missing_audio > 0:
        issues.append(
            {
                "severity": "warn",
                "code": "source.audio_missing",
                "message": f"{dataset_id}: rows with missing audio files = {missing_audio}",
            }
        )

    coverages = {field: round(_coverage(rows, field), 6) for field in OPTIONAL_AUDIT_FIELDS if field in fields}
    source_dataset_values = Counter(
        str(r.get("source_dataset", "")).strip() for r in rows if str(r.get("source_dataset", "")).strip()
    )

    return {
        "dataset_id": dataset_id,
        "culture": culture,
        "local_metadata": str(local_metadata),
        "exists": True,
        "n_rows": int(len(rows)),
        "fieldnames": fieldnames,
        "duplicate_track_ids": duplicate_track_ids,
        "missing_audio_rows": int(missing_audio),
        "culture_values": dict(sorted(culture_values.items())),
        "source_dataset_values": dict(sorted(source_dataset_values.items())),
        "coverage": coverages,
    }, issues


def _audit_embedding_config(name: str, cfg: Any) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if not isinstance(cfg, dict):
        issues.append(
            {
                "severity": "error",
                "code": "manifest.embedding_invalid",
                "message": f"embedding '{name}' config must be a JSON object",
            }
        )
        return {"name": name, "enabled": False}, issues

    enabled = bool(cfg.get("enabled", False))
    row = {
        "name": name,
        "enabled": enabled,
        "model_id": cfg.get("model_id"),
        "window_count": cfg.get("window_count"),
        "window_strategy": cfg.get("window_strategy"),
        "window_aggregate": cfg.get("window_aggregate"),
    }
    if not enabled:
        return row, issues

    required = REQUIRED_EMBEDDING_FIELDS.get(name, set())
    missing = sorted(key for key in required if key not in cfg)
    if missing:
        issues.append(
            {
                "severity": "error",
                "code": "manifest.embedding_missing_keys",
                "message": f"embedding '{name}' missing required keys: {missing}",
            }
        )

    if "layer_mode" in cfg:
        issues.append(
            {
                "severity": "warn",
                "code": "manifest.embedding_legacy_layer_mode",
                "message": f"embedding '{name}' still uses legacy key 'layer_mode'; use layer_indices/layer_weights instead",
            }
        )

    if name == "culturemert":
        layer_indices = cfg.get("layer_indices")
        layer_weights = cfg.get("layer_weights")
        if layer_weights is not None and layer_indices is None:
            issues.append(
                {
                    "severity": "error",
                    "code": "manifest.embedding_layer_weights_without_indices",
                    "message": "culturemert.layer_weights requires culturemert.layer_indices",
                }
            )
        if (
            isinstance(layer_indices, list)
            and isinstance(layer_weights, list)
            and len(layer_indices) != len(layer_weights)
        ):
            issues.append(
                {
                    "severity": "error",
                    "code": "manifest.embedding_layer_mismatch",
                    "message": "culturemert.layer_indices and culturemert.layer_weights must have the same length",
                }
            )

    return row, issues


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    issues: list[dict[str, str]] = []
    source_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []

    required_top = {
        "dataset_name",
        "dataset_version",
        "schema_version",
        "root_out_dir",
        "sources",
    }
    missing_top = sorted(required_top - set(manifest.keys()))
    if missing_top:
        issues.append(
            {
                "severity": "error",
                "code": "manifest.top_level_missing",
                "message": f"manifest missing top-level keys: {missing_top}",
            }
        )

    missing_recommended = sorted(RECOMMENDED_TOP_LEVEL - set(manifest.keys()))
    if missing_recommended:
        issues.append(
            {
                "severity": "info",
                "code": "manifest.recommended_top_level_missing",
                "message": f"manifest is missing recommended keys: {missing_recommended}",
            }
        )

    sources = manifest.get("sources", [])
    if not isinstance(sources, list) or not sources:
        issues.append(
            {
                "severity": "error",
                "code": "manifest.sources_empty",
                "message": "manifest sources must be a non-empty list",
            }
        )
        sources = []

    cultures = Counter()
    dataset_ids = Counter()
    for source in sources:
        if not isinstance(source, dict):
            issues.append(
                {
                    "severity": "error",
                    "code": "manifest.source_invalid",
                    "message": "manifest source entry is not a JSON object",
                }
            )
            continue
        dataset_id = str(source.get("dataset_id", "")).strip()
        if dataset_id == "":
            issues.append(
                {
                    "severity": "error",
                    "code": "manifest.source_dataset_id_missing",
                    "message": "manifest source entry is missing dataset_id",
                }
            )
        else:
            dataset_ids[dataset_id] += 1
        culture = str(source.get("culture", "")).strip()
        if culture == "":
            issues.append(
                {
                    "severity": "error",
                    "code": "manifest.source_culture_missing",
                    "message": f"{dataset_id or '<unknown>'}: culture is missing",
                }
            )
        else:
            cultures[culture] += 1

        row, row_issues = _audit_source(source)
        source_rows.append(row)
        issues.extend(row_issues)

    duplicate_dataset_ids = sorted([dataset_id for dataset_id, cnt in dataset_ids.items() if cnt > 1])
    if duplicate_dataset_ids:
        issues.append(
            {
                "severity": "warn",
                "code": "manifest.duplicate_dataset_id",
                "message": f"duplicate dataset_id values: {duplicate_dataset_ids}",
            }
        )

    if any(cnt == 1 for cnt in cultures.values()):
        risky = sorted([culture for culture, cnt in cultures.items() if cnt == 1])
        issues.append(
            {
                "severity": "info",
                "code": "manifest.single_source_cultures",
                "message": f"cultures backed by a single manifest source: {', '.join(risky)}",
            }
        )

    embeddings = manifest.get("embeddings", {})
    if isinstance(embeddings, dict):
        for name in sorted(embeddings.keys()):
            row, row_issues = _audit_embedding_config(name, embeddings.get(name))
            embedding_rows.append(row)
            issues.extend(row_issues)
    elif "embeddings" in manifest:
        issues.append(
            {
                "severity": "error",
                "code": "manifest.embeddings_invalid",
                "message": "manifest.embeddings must be a JSON object",
            }
        )

    validation = manifest.get("validation", {})
    if isinstance(validation, dict):
        for key in [
            "min_tracks_per_culture",
            "max_culture_imbalance_ratio",
            "max_unknown_track_ratio",
            "max_duplicate_user_track_ratio",
            "max_zero_norm_ratio",
            "min_interactions_per_user",
        ]:
            if key not in validation:
                issues.append(
                    {
                        "severity": "info",
                        "code": "manifest.validation_missing_key",
                        "message": f"validation is missing recommended key '{key}'",
                    }
                )
    elif "validation" in manifest:
        issues.append(
            {
                "severity": "error",
                "code": "manifest.validation_invalid",
                "message": "manifest.validation must be a JSON object",
            }
        )

    return {
        "manifest": _safe_rel(manifest_path),
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("dataset_version"),
        "schema_version": manifest.get("schema_version"),
        "root_out_dir": manifest.get("root_out_dir"),
        "n_sources": len(source_rows),
        "cultures": dict(sorted(cultures.items())),
        "sources": source_rows,
        "embeddings": embedding_rows,
        "issues": issues,
        "embedding_keys": sorted(list((manifest.get("embeddings") or {}).keys()))
        if isinstance(manifest.get("embeddings"), dict)
        else [],
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.extend(["# Dataset Manifest Audit", ""])
    lines.append(f"- manifest: `{report['manifest']}`")
    lines.append(f"- dataset_name: `{report.get('dataset_name')}`")
    lines.append(f"- dataset_version: `{report.get('dataset_version')}`")
    lines.append(f"- schema_version: `{report.get('schema_version')}`")
    lines.append(f"- sources: `{int(report.get('n_sources', 0))}`")
    lines.append("")

    lines.extend(["## Cultures", "", "| culture | manifest_source_count |", "|---|---:|"])
    for culture, cnt in dict(report.get("cultures", {})).items():
        lines.append(f"| {culture} | {cnt} |")

    lines.extend(
        [
            "",
            "## Source Audit",
            "",
            "| dataset_id | culture | exists | rows | duplicate_track_ids | missing_audio_rows |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in report.get("sources", []):
        lines.append(
            f"| {row.get('dataset_id', '')} | {row.get('culture', '')} | {str(bool(row.get('exists'))).lower()} | "
            f"{int(row.get('n_rows', 0) or 0)} | {int(row.get('duplicate_track_ids', 0) or 0)} | {int(row.get('missing_audio_rows', 0) or 0)} |"
        )

    lines.extend(
        [
            "",
            "## Embeddings",
            "",
            "| name | enabled | model_id | window_count | window_strategy | window_aggregate |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in report.get("embeddings", []):
        lines.append(
            f"| {row.get('name', '')} | {str(bool(row.get('enabled'))).lower()} | {row.get('model_id', '')} | "
            f"{int(row.get('window_count', 0) or 0)} | {row.get('window_strategy', '')} | {row.get('window_aggregate', '')} |"
        )

    lines.extend(["", "## Issues", ""])
    issues = list(report.get("issues", []))
    if not issues:
        lines.append("- none")
    else:
        lines.append("| severity | code | message |")
        lines.append("|---|---|---|")
        for issue in issues:
            lines.append(f"| {issue['severity']} | {issue['code']} | {issue['message']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a dataset manifest and its source metadata files.")
    ap.add_argument("--manifest", required=True, help="Path to the dataset manifest JSON")
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "reports" / "datasets" / "manifest_audit"),
        help="Directory for markdown/json outputs",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = audit_manifest(manifest_path)
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_to_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"out_dir": str(out_dir), "issues": len(report.get("issues", []))},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
