from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torchaudio


def check_prereqs(
    metadata_csv: str,
    track_id_col: str = "track_id",
    audio_col: str = "audio_path",
    sample_size: int = 20,
) -> dict[str, Any]:
    p = Path(metadata_csv)
    if not p.exists():
        raise FileNotFoundError(f"metadata not found: {p}")

    rows: list[dict[str, str]] = []
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if track_id_col not in cols:
            raise ValueError(f"missing track_id column: {track_id_col}")
        if audio_col not in cols:
            raise ValueError(f"missing audio path column: {audio_col}")
        for r in reader:
            rows.append({str(k): str(v) for k, v in r.items()})

    total = len(rows)
    missing_audio = 0
    missing_files = 0
    existing = 0
    failed_info = 0
    sample_checked = 0
    sample_rates: list[int] = []
    durations: list[float] = []
    bad_examples: list[dict[str, str]] = []

    for i, r in enumerate(rows):
        _ = i
        tid = str(r.get(track_id_col, "")).strip()
        ap = str(r.get(audio_col, "")).strip()
        if not ap:
            missing_audio += 1
            if len(bad_examples) < 20:
                bad_examples.append({"track_id": tid, "issue": "empty_audio_path"})
            continue
        fp = Path(ap)
        if not fp.exists():
            missing_files += 1
            if len(bad_examples) < 20:
                bad_examples.append({"track_id": tid, "issue": "missing_file", "path": ap})
            continue
        existing += 1
        if sample_checked >= int(sample_size):
            continue
        sample_checked += 1
        try:
            info = torchaudio.info(str(fp))
            sr = int(info.sample_rate)
            n = int(info.num_frames)
            sample_rates.append(sr)
            if sr > 0 and n > 0:
                durations.append(float(n / sr))
        except Exception:
            failed_info += 1
            if len(bad_examples) < 20:
                bad_examples.append({"track_id": tid, "issue": "torchaudio_info_failed", "path": ap})

    status = "pass"
    if missing_audio > 0 or missing_files > 0:
        status = "warn"
    if total == 0:
        status = "fail"

    summary = {
        "status": status,
        "metadata_csv": str(p),
        "track_id_col": str(track_id_col),
        "audio_col": str(audio_col),
        "n_rows": int(total),
        "n_missing_audio_path": int(missing_audio),
        "n_missing_files": int(missing_files),
        "n_existing_files": int(existing),
        "sample_size": int(sample_checked),
        "n_info_failures": int(failed_info),
        "sample_rate_min": int(min(sample_rates)) if sample_rates else None,
        "sample_rate_max": int(max(sample_rates)) if sample_rates else None,
        "duration_min_sec": float(min(durations)) if durations else None,
        "duration_max_sec": float(max(durations)) if durations else None,
        "duration_mean_sec": float(sum(durations) / len(durations)) if durations else None,
        "bad_examples": bad_examples,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Check prerequisites for waveform generation pipeline.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--track_id_col", default="track_id")
    ap.add_argument("--audio_col", default="audio_path")
    ap.add_argument("--sample_size", type=int, default=20)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rep = check_prereqs(
        metadata_csv=str(args.metadata),
        track_id_col=str(args.track_id_col),
        audio_col=str(args.audio_col),
        sample_size=int(args.sample_size),
    )
    if args.out_json:
        out = Path(str(args.out_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()

