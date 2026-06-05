from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from dcas.scripts.align_assets_to_tracks import align_assets_to_tracks
from dcas.scripts.audit_dataset_manifest import audit_manifest
from dcas.scripts.audit_dataset_v4 import MetadataAuditThresholds, audit_dataset_v4
from dcas.scripts.build_tracks_from_audio import build_tracks_from_audio
from dcas.scripts.build_tracks_with_gemini import build_tracks_with_gemini
from dcas.scripts.harmonize_v4_metadata import harmonize_v4_metadata
from dcas.scripts.merge_metadata_dedup import merge_metadata_dedup
from dcas.scripts.synthesize_interactions import synthesize_interactions
from dcas.scripts.validate_dataset import ValidationThresholds, validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGES = ["merge", "harmonize", "interactions", "audit"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _metadata_thresholds(manifest: dict[str, Any]) -> MetadataAuditThresholds:
    validation = manifest.get("validation", {})
    return MetadataAuditThresholds(
        min_tracks_per_culture=int(validation.get("min_tracks_per_culture", 30)),
        max_culture_imbalance_ratio=float(validation.get("max_culture_imbalance_ratio", 20.0)),
        min_interactions_per_user=int(validation.get("min_interactions_per_user", 5)),
        max_unknown_track_ratio=float(validation.get("max_unknown_track_ratio", 0.01)),
        max_duplicate_user_track_ratio=float(validation.get("max_duplicate_user_track_ratio", 0.05)),
    )


def _track_validation_thresholds(manifest: dict[str, Any]) -> ValidationThresholds:
    validation = manifest.get("validation", {})
    return ValidationThresholds(
        min_tracks_per_culture=int(validation.get("min_tracks_per_culture", 30)),
        max_culture_imbalance_ratio=float(validation.get("max_culture_imbalance_ratio", 20.0)),
        max_unknown_track_ratio=float(validation.get("max_unknown_track_ratio", 0.01)),
        max_duplicate_user_track_ratio=float(validation.get("max_duplicate_user_track_ratio", 0.05)),
        max_zero_norm_ratio=float(validation.get("max_zero_norm_ratio", 0.05)),
        min_interactions_per_user=int(validation.get("min_interactions_per_user", 5)),
    )


def _interaction_outputs(out_root: Path) -> dict[str, Path]:
    return {
        "single": out_root / "interactions_synth_single.csv",
        "mixed": out_root / "interactions_synth_mixed.csv",
    }


def _enabled_interactions(manifest: dict[str, Any], out_root: Path) -> list[Path]:
    protocol = manifest.get("interaction_protocol", {})
    outputs = _interaction_outputs(out_root)
    out: list[Path] = []
    for key in ["single", "mixed"]:
        cfg = protocol.get(key, {})
        if bool(cfg.get("enabled", False)) and outputs[key].exists():
            out.append(outputs[key])
    return out


def _embedding_track_name(name: str, cfg: dict[str, Any]) -> str:
    window_count = int(cfg.get("window_count", 1))
    if name == "culturemert":
        return f"tracks_culturemert_mw{window_count}.npz"
    if name == "gemini":
        model_id = str(cfg.get("model_id", "gemini")).replace("/", "_").replace("-", "_")
        if "embedding_2" in model_id:
            model_stub = "gemini_embedding2"
        else:
            model_stub = model_id
        return f"tracks_{model_stub}_mw{window_count}.npz"
    return f"tracks_{name}_mw{window_count}.npz"


def _build_culturemert(metadata_csv: Path, out_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out_npz = out_root / _embedding_track_name("culturemert", cfg)
    return build_tracks_from_audio(
        metadata_csv=metadata_csv,
        out_npz=out_npz,
        model_id=str(cfg.get("model_id", "ntua-slp/CultureMERT-95M")),
        device=str(cfg.get("device")) if cfg.get("device") is not None else None,
        pooling=str(cfg.get("pooling", "mean")),
        layer_indices=list(cfg.get("layer_indices")) if cfg.get("layer_indices") is not None else None,
        layer_weights=list(cfg.get("layer_weights")) if cfg.get("layer_weights") is not None else None,
        max_seconds=float(cfg.get("max_seconds", 30.0)) if cfg.get("max_seconds") is not None else None,
        window_count=int(cfg.get("window_count", 1)),
        window_strategy=str(cfg.get("window_strategy", "single")),
        window_aggregate=str(cfg.get("window_aggregate", "mean")),
        limit=int(cfg["limit"]) if cfg.get("limit") is not None else None,
        skip_errors=bool(cfg.get("skip_errors", False)),
    )


def _build_gemini(metadata_csv: Path, out_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out_npz = out_root / _embedding_track_name("gemini", cfg)
    return build_tracks_with_gemini(
        metadata_csv=metadata_csv,
        out_npz=out_npz,
        model_id=str(cfg.get("model_id", "gemini-embedding-2-preview")),
        api_key=str(cfg.get("api_key")) if cfg.get("api_key") is not None else None,
        api_key_file=str(cfg.get("api_key_file")) if cfg.get("api_key_file") is not None else None,
        vertexai=bool(cfg.get("vertexai", False)),
        vertex_project=str(cfg.get("vertex_project")) if cfg.get("vertex_project") is not None else None,
        vertex_location=str(cfg.get("vertex_location")) if cfg.get("vertex_location") is not None else None,
        output_dimensionality=int(cfg.get("output_dimensionality", 768)),
        task_type=str(cfg.get("task_type")) if cfg.get("task_type") is not None else None,
        max_seconds=float(cfg.get("max_seconds", 30.0)) if cfg.get("max_seconds") is not None else None,
        target_sample_rate=int(cfg.get("target_sample_rate", 16000)),
        window_count=int(cfg.get("window_count", 1)),
        window_strategy=str(cfg.get("window_strategy", "single")),
        window_aggregate=str(cfg.get("window_aggregate", "mean")),
        limit=int(cfg["limit"]) if cfg.get("limit") is not None else None,
        skip_errors=bool(cfg.get("skip_errors", False)),
        cache_dir=str(cfg.get("cache_dir")) if cfg.get("cache_dir") is not None else None,
        dry_run=bool(cfg.get("dry_run", False)),
        max_workers=int(cfg.get("max_workers", 1)),
    )


def build_research_dataset_v4(
    manifest_path: str | Path,
    stages: list[str] | None = None,
    embedding_targets: list[str] | None = None,
    allow_manifest_errors: bool = False,
) -> dict[str, Any]:
    manifest_file = _resolve_path(manifest_path)
    manifest = _load_json(manifest_file)
    stage_list = [str(stage).strip().lower() for stage in (stages or DEFAULT_STAGES)]
    out_root = _resolve_path(str(manifest["root_out_dir"]))
    dataset_key = out_root.name
    reports_root = REPO_ROOT / "reports" / "datasets" / str(manifest["dataset_name"]) / dataset_key
    out_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    selected_embedding_targets = {
        str(name).strip().lower() for name in (embedding_targets or []) if str(name).strip() != ""
    }

    manifest_report = audit_manifest(manifest_file)
    manifest_audit_dir = reports_root / "manifest_audit"
    manifest_audit_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_audit_dir / "summary.json", manifest_report)
    (manifest_audit_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Dataset Manifest Audit",
                "",
                f"- manifest: `{manifest_file}`",
                f"- issues: `{len(manifest_report.get('issues', []))}`",
            ]
        ),
        encoding="utf-8",
    )
    if not allow_manifest_errors and any(issue["severity"] == "error" for issue in manifest_report.get("issues", [])):
        raise RuntimeError(f"manifest audit has blocking errors: {manifest_file}")

    _write_json(out_root / "manifest.snapshot.json", manifest)

    metadata_raw = out_root / "metadata_raw.csv"
    metadata_clean = out_root / "metadata_clean.csv"
    metadata_harmonized = out_root / "metadata_harmonized.csv"
    metadata_release = out_root / "metadata_release.csv"
    interaction_outputs = _interaction_outputs(out_root)

    build_report: dict[str, Any] = {
        "manifest": str(manifest_file),
        "out_root": str(out_root),
        "reports_root": str(reports_root),
        "stages": stage_list,
        "steps": {},
    }

    if "merge" in stage_list:
        source_paths = [_resolve_path(str(source["local_metadata"])) for source in manifest.get("sources", [])]
        build_report["steps"]["merge"] = merge_metadata_dedup(inputs=source_paths, out_csv=metadata_raw)

    if "harmonize" in stage_list:
        build_report["steps"]["harmonize"] = harmonize_v4_metadata(
            metadata_csv=metadata_raw,
            out_clean_csv=metadata_clean,
            out_harmonized_csv=metadata_harmonized,
            dataset_version=str(manifest["dataset_version"]),
            schema_version=str(manifest["schema_version"]),
            import_batch=str(manifest.get("dataset_version", "")),
        )
        shutil.copyfile(metadata_harmonized, metadata_release)

    if "interactions" in stage_list:
        protocol = manifest.get("interaction_protocol", {})
        interaction_steps: dict[str, Any] = {}
        for key, out_path in interaction_outputs.items():
            cfg = protocol.get(key, {})
            if not bool(cfg.get("enabled", False)):
                continue
            interaction_steps[key] = synthesize_interactions(
                metadata_csv=metadata_release,
                out_csv=out_path,
                users_per_culture=int(cfg.get("users_per_culture", 20)),
                tracks_per_user=int(cfg.get("tracks_per_user", 50)),
                min_weight=float(cfg.get("min_weight", 0.5)),
                max_weight=float(cfg.get("max_weight", 2.0)),
                genre_column=str(cfg.get("genre_column", "coarse_label")),
                mode=str(cfg.get("mode", "single_culture")),
                secondary_cultures=int(cfg.get("secondary_cultures", 2)),
                home_share=float(cfg.get("home_share", 0.65)),
                seed=int(cfg.get("seed", 42)),
            )
        build_report["steps"]["interactions"] = interaction_steps

    if "audit" in stage_list:
        dataset_profile = audit_dataset_v4(
            metadata_csv=metadata_release,
            out_dir=reports_root,
            interactions=_enabled_interactions(manifest, out_root),
            dataset_name=f"{manifest['dataset_name']}::{dataset_key}",
            thresholds=_metadata_thresholds(manifest),
        )
        build_report["steps"]["audit"] = {
            "issues": int(len(dataset_profile.get("issues", []))),
            "dataset_profile_json": str((reports_root / "dataset_profile.json").resolve()),
        }
        validation_report = {
            "dataset_name": manifest["dataset_name"],
            "dataset_version": manifest["dataset_version"],
            "schema_version": manifest["schema_version"],
            "profile_summary": {
                "n_rows": dataset_profile["profile"]["n_rows"],
                "n_cultures": dataset_profile["profile"]["n_cultures"],
                "n_sources": dataset_profile["profile"]["n_sources"],
            },
            "issues": dataset_profile["issues"],
        }
        _write_json(out_root / "validation_report.json", validation_report)
        data_card = {
            "dataset_name": manifest["dataset_name"],
            "dataset_version": manifest["dataset_version"],
            "schema_version": manifest["schema_version"],
            "root_out_dir": str(out_root),
            "sources": manifest.get("sources", []),
            "profile": dataset_profile["profile"],
            "source_confound": dataset_profile["source_confound"],
            "interactions": dataset_profile["interactions"],
            "planned_embeddings": manifest.get("embeddings", {}),
        }
        _write_json(out_root / "data_card.json", data_card)

    if "embeddings" in stage_list:
        embedding_steps: dict[str, Any] = {}
        validation_thresholds = _track_validation_thresholds(manifest)
        mixed_interactions = interaction_outputs["mixed"] if interaction_outputs["mixed"].exists() else None
        for name, cfg in sorted((manifest.get("embeddings") or {}).items()):
            if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
                continue
            if selected_embedding_targets and str(name).strip().lower() not in selected_embedding_targets:
                continue
            if name == "culturemert":
                embed_report = _build_culturemert(metadata_release, out_root, cfg)
            elif name == "gemini":
                embed_report = _build_gemini(metadata_release, out_root, cfg)
            else:
                continue

            if not embed_report.get("out"):
                embedding_steps[name] = embed_report
                continue

            tracks_path = Path(str(embed_report["out"]))
            aligned_metadata = metadata_release
            aligned_interactions = mixed_interactions
            if int(embed_report.get("n_tracks", 0)) < _count_csv_rows(metadata_release):
                suffix = tracks_path.stem.replace("tracks_", "")
                aligned_metadata = out_root / f"metadata_release_{suffix}.csv"
                aligned_interactions = (
                    (out_root / f"interactions_synth_mixed_{suffix}.csv") if mixed_interactions else None
                )
                align_report = align_assets_to_tracks(
                    tracks_path=tracks_path,
                    metadata_in=metadata_release,
                    metadata_out=aligned_metadata,
                    interactions_in=mixed_interactions,
                    interactions_out=aligned_interactions,
                )
                embed_report["alignment"] = align_report

            validation_report = validate_dataset(
                tracks_path=tracks_path,
                interactions_path=aligned_interactions,
                thresholds=validation_thresholds,
            )
            validation_dir = reports_root / f"validate_{name}"
            validation_dir.mkdir(parents=True, exist_ok=True)
            _write_json(validation_dir / "report.json", validation_report)
            (validation_dir / "report.md").write_text(
                "\n".join(
                    [
                        "# Dataset Validation",
                        "",
                        f"- status: `{validation_report['status']}`",
                        f"- tracks: `{validation_report['summary']['n_tracks']}`",
                        f"- interactions: `{validation_report['summary']['n_interactions']}`",
                        f"- issues: `{validation_report['summary']['n_issues']}`",
                    ]
                ),
                encoding="utf-8",
            )
            embed_report["validation"] = {
                "status": validation_report["status"],
                "report_json": str((validation_dir / "report.json").resolve()),
                "report_md": str((validation_dir / "report.md").resolve()),
                "summary": validation_report["summary"],
            }
            embedding_steps[name] = embed_report
        build_report["steps"]["embeddings"] = embedding_steps

    _write_json(reports_root / "build_report.json", build_report)
    return build_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a V4 dataset skeleton from a manifest.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--stages",
        nargs="*",
        default=DEFAULT_STAGES,
        choices=["merge", "harmonize", "interactions", "audit", "embeddings"],
    )
    ap.add_argument(
        "--embedding_targets",
        nargs="*",
        default=None,
        help="Optional subset of embedding builders to run during the embeddings stage, e.g. culturemert gemini",
    )
    ap.add_argument("--allow_manifest_errors", action="store_true")
    args = ap.parse_args()

    report = build_research_dataset_v4(
        manifest_path=args.manifest,
        stages=list(args.stages or DEFAULT_STAGES),
        embedding_targets=list(args.embedding_targets or []),
        allow_manifest_errors=bool(args.allow_manifest_errors),
    )
    print(
        json.dumps(
            {
                "reports_root": report["reports_root"],
                "steps": sorted(report["steps"].keys()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
