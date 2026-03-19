from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


YAMBDA_REPO = "yandex/yambda"
YAMBDA_FILES = (
    "sequential/50m/multi_event.parquet",
    "sequential/500m/multi_event.parquet",
    "sequential/5b/multi_event.parquet",
    "embeddings.parquet",
)


def _yambda_probe() -> dict[str, Any]:
    api = HfApi()
    tree = api.list_repo_tree(YAMBDA_REPO, repo_type="dataset", recursive=True)
    sizes: dict[str, int | None] = {name: None for name in YAMBDA_FILES}
    for node in tree:
        path = getattr(node, "path", "")
        if path in sizes:
            sizes[path] = getattr(node, "size", None)
    return {
        "repo": YAMBDA_REPO,
        "access": "public",
        "subset_rows": {
            "sequential_50m_users": 10_000,
            "sequential_500m_users": 100_000,
            "sequential_5b_users": 1_000_000,
        },
        "key_file_sizes_bytes": sizes,
        "schema_focus": {
            "interaction_fields": [
                "uid",
                "item_id",
                "timestamp",
                "is_organic",
                "event_type",
                "played_ratio_pct",
                "track_length_seconds",
            ],
            "embedding_fields": ["item_id", "embed", "normalized_embed"],
            "missing_for_current_repo": ["culture"],
        },
    }


def _mssd_probe() -> dict[str, Any]:
    return {
        "site": "https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge",
        "publication": "https://research.atspotify.com/publications/the-music-streaming-sessions-dataset-short-paper/",
        "access": "not_publicly_downloadable",
        "note": "AIcrowd challenge page says that, since 2024-07-08, the dataset is no longer available for download and access requires contacting Spotify Research.",
        "task": "session-based sequential skip prediction",
        "public_dataset_scale": {
            "sessions": 130_000_000,
            "tracks": 4_000_000,
        },
        "missing_for_current_repo": ["culture", "target_culture offline protocol"],
    }


def _repo_constraints() -> dict[str, Any]:
    return {
        "current_tracks_format": {
            "required_fields": ["track_id", "culture", "embedding"],
            "optional_fields": ["source_dataset", "affect_label"],
        },
        "current_interactions_format": {
            "required_fields": ["user_id", "track_id", "weight"],
        },
        "current_benchmark_assumptions": [
            "recommendation is evaluated against a target culture per user",
            "cultural calibration and target-culture probability require track-level culture labels",
            "current BPR trainer materializes per-user negative pools, which does not scale to industrial item counts",
        ],
        "recommended_next_step": "Add a separate scalable recsys benchmark path for Recall/NDCG/MRR before trying Yambda or MSSD.",
    }


def probe_external_benchmarks(out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "yambda": _yambda_probe(),
        "mssd": _mssd_probe(),
        "repo_constraints": _repo_constraints(),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "report_json": str(out_path.resolve()),
        "yambda_repo": YAMBDA_REPO,
        "mssd_site": report["mssd"]["site"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe public recsys benchmarks against the current repository assumptions.")
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "reports/external_benchmarks/public_benchmark_probe_2026-03-19.json"),
        help="Output JSON report path.",
    )
    args = ap.parse_args()
    rep = probe_external_benchmarks(out_path=Path(str(args.out)))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
