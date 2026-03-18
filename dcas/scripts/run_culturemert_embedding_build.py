from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.scripts.build_tracks_from_audio import build_tracks_from_audio


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CultureMERT embedding build from a JSON config file.")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    args = ap.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    out = build_tracks_from_audio(
        metadata_csv=cfg["metadata"],
        out_npz=cfg["out"],
        model_id=cfg.get("model_id", "ntua-slp/CultureMERT-95M"),
        device=cfg.get("device"),
        pooling=str(cfg.get("pooling", "mean")),
        max_seconds=cfg.get("max_seconds", 30.0),
        window_count=int(cfg.get("window_count", 1)),
        window_strategy=str(cfg.get("window_strategy", "single")),
        window_aggregate=str(cfg.get("window_aggregate", "mean")),
        limit=cfg.get("limit"),
        skip_errors=bool(cfg.get("skip_errors", False)),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
