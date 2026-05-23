from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dcas.pal.wording import PAL_README_REMINDER_EN
from dcas.pipelines import pal_tasks
from dcas.scripts.export_pal_annotation_sheet import export_pal_annotation_sheet
from dcas.scripts.select_pal_tasks_stratified import select_pal_tasks_stratified


def _read_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_readme(path: Path, manifest: dict[str, Any]) -> None:
    files = manifest["files"]
    lines = [
        "# V4 Real PAL Bundle",
        "",
        "This folder is ready for human PAL collection.",
        "",
        "Recommended order:",
        "1. Start with the pilot sheet and verify annotator understanding.",
        "2. Revise instructions if the pilot reveals ambiguity.",
        "3. Move to the round-1 sheet for the main annotation batch.",
        "4. Save the completed CSV as tasks_round1_200_annotation_filled.csv in this folder.",
        "5. Run run_pal_platform with pal_v4_main_culturemert_real.run.json.",
        "",
        "Generated files:",
        f"- candidates: {files['candidate_tasks']}",
        f"- candidate annotation sheet: {files['candidate_sheet']}",
        f"- pilot tasks: {files['pilot_tasks']}",
        f"- pilot annotation sheet: {files['pilot_sheet']}",
        f"- round-1 tasks: {files['round1_tasks']}",
        f"- round-1 annotation sheet: {files['round1_sheet']}",
        f"- manifest: {path.parent / 'bundle_manifest.json'}",
        "",
        "Annotation reminder:",
        f"- {PAL_README_REMINDER_EN}",
        "- Do not decide directly from culture labels, language names, or source names.",
        "- Fill `similar` with yes/no (or 1/0) and add one short rationale.",
        "- Leave difficult cases blank and explain the reason in `notes`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_real_pal_bundle(
    tracks_path: str,
    metadata_csv: str,
    baseline_model_path: str,
    out_dir: str,
    candidate_tasks: int = 1000,
    pilot_tasks: int = 20,
    round1_tasks: int = 200,
    group_field: str = "culture",
    pool_multiplier: int = 3,
    uncertainty_method: str = "culture_centroid_entropy",
    pilot_seed: int = 42,
    round1_seed: int = 43,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    candidate_tasks_path = out_dir_p / f"candidates_{int(candidate_tasks)}.jsonl"
    candidate_sheet_path = out_dir_p / f"candidates_{int(candidate_tasks)}_annotation.csv"
    pilot_tasks_path = out_dir_p / f"pilot_tasks_{int(pilot_tasks)}.jsonl"
    pilot_sheet_path = out_dir_p / f"pilot_tasks_{int(pilot_tasks)}_annotation.csv"
    round1_tasks_path = out_dir_p / f"tasks_round1_{int(round1_tasks)}.jsonl"
    round1_sheet_path = out_dir_p / f"tasks_round1_{int(round1_tasks)}_annotation.csv"

    candidate_info = pal_tasks(
        model_path=str(baseline_model_path),
        tracks_path=str(tracks_path),
        out_path=candidate_tasks_path,
        n=int(candidate_tasks),
        prefer_cuda=bool(prefer_cuda),
        uncertainty_method=str(uncertainty_method),
    )
    candidate_sheet_info = export_pal_annotation_sheet(
        tasks_path=str(candidate_tasks_path),
        metadata_csv=str(metadata_csv),
        out_csv=str(candidate_sheet_path),
    )

    pilot_info = select_pal_tasks_stratified(
        tasks_path=str(candidate_tasks_path),
        out_path=str(pilot_tasks_path),
        n_total=int(pilot_tasks),
        group_field=str(group_field),
        pool_multiplier=int(pool_multiplier),
        seed=int(pilot_seed),
    )
    pilot_sheet_info = export_pal_annotation_sheet(
        tasks_path=str(pilot_tasks_path),
        metadata_csv=str(metadata_csv),
        out_csv=str(pilot_sheet_path),
    )

    round1_info = select_pal_tasks_stratified(
        tasks_path=str(candidate_tasks_path),
        out_path=str(round1_tasks_path),
        n_total=int(round1_tasks),
        group_field=str(group_field),
        pool_multiplier=int(pool_multiplier),
        seed=int(round1_seed),
    )
    round1_sheet_info = export_pal_annotation_sheet(
        tasks_path=str(round1_tasks_path),
        metadata_csv=str(metadata_csv),
        out_csv=str(round1_sheet_path),
    )

    manifest = {
        "tracks_path": str(Path(tracks_path).resolve()),
        "metadata_csv": str(Path(metadata_csv).resolve()),
        "baseline_model_path": str(Path(baseline_model_path).resolve()),
        "out_dir": str(out_dir_p.resolve()),
        "uncertainty_method": str(uncertainty_method),
        "group_field": str(group_field),
        "pool_multiplier": int(pool_multiplier),
        "counts": {
            "candidate_tasks": int(candidate_tasks),
            "pilot_tasks": int(pilot_tasks),
            "round1_tasks": int(round1_tasks),
        },
        "files": {
            "candidate_tasks": str(candidate_tasks_path.resolve()),
            "candidate_sheet": str(candidate_sheet_path.resolve()),
            "pilot_tasks": str(pilot_tasks_path.resolve()),
            "pilot_sheet": str(pilot_sheet_path.resolve()),
            "round1_tasks": str(round1_tasks_path.resolve()),
            "round1_sheet": str(round1_sheet_path.resolve()),
        },
        "reports": {
            "candidate_info": candidate_info,
            "candidate_sheet_info": candidate_sheet_info,
            "pilot_info": pilot_info,
            "pilot_sheet_info": pilot_sheet_info,
            "round1_info": round1_info,
            "round1_sheet_info": round1_sheet_info,
        },
        "next_steps": [
            "Use the pilot annotation sheet first.",
            "Revise instructions if the pilot reveals ambiguity.",
            "Collect the round-1 human PAL sheet.",
            "Save the completed sheet as tasks_round1_200_annotation_filled.csv.",
            "Run the real PAL config with run_pal_platform.",
        ],
    }

    _write_json(out_dir_p / "bundle_manifest.json", manifest)
    _write_readme(out_dir_p / "README.md", manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a real-PAL bundle with candidates, pilot tasks, and round-1 sheets.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _read_config(str(args.config))
    out = prepare_real_pal_bundle(
        tracks_path=str(cfg["tracks"]),
        metadata_csv=str(cfg["metadata"]),
        baseline_model_path=str(cfg["baseline_model"]),
        out_dir=str(cfg["out_dir"]),
        candidate_tasks=int(cfg.get("candidate_tasks", 1000)),
        pilot_tasks=int(cfg.get("pilot_tasks", 20)),
        round1_tasks=int(cfg.get("round1_tasks", 200)),
        group_field=str(cfg.get("group_field", "culture")),
        pool_multiplier=int(cfg.get("pool_multiplier", 3)),
        uncertainty_method=str(cfg.get("uncertainty_method", "culture_centroid_entropy")),
        pilot_seed=int(cfg.get("pilot_seed", 42)),
        round1_seed=int(cfg.get("round1_seed", 43)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
