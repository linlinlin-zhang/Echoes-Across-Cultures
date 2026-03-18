from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.scripts.build_tracks_with_gemini import build_tracks_with_gemini


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Gemini embedding build from a JSON config file.")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    args = ap.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    out = build_tracks_with_gemini(
        metadata_csv=cfg["metadata"],
        out_npz=cfg["out"],
        model_id=cfg.get("model_id", "gemini-embedding-2-preview"),
        api_key=cfg.get("api_key"),
        api_key_file=cfg.get("api_key_file"),
        vertexai=bool(cfg.get("vertexai", False)),
        vertex_project=cfg.get("vertex_project"),
        vertex_location=cfg.get("vertex_location"),
        output_dimensionality=int(cfg.get("output_dimensionality", 768)),
        task_type=cfg.get("task_type"),
        max_seconds=cfg.get("max_seconds", 30.0),
        target_sample_rate=int(cfg.get("target_sample_rate", 16000)),
        window_count=int(cfg.get("window_count", 1)),
        window_strategy=str(cfg.get("window_strategy", "single")),
        window_aggregate=str(cfg.get("window_aggregate", "mean")),
        limit=cfg.get("limit"),
        skip_errors=bool(cfg.get("skip_errors", False)),
        cache_dir=cfg.get("cache_dir"),
        dry_run=bool(cfg.get("dry_run", False)),
        max_workers=int(cfg.get("max_workers", 1)),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
