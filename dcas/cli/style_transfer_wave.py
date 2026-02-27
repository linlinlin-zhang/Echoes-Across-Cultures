from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from dcas.waveform_style_transfer import transfer_waveform_style


def _read_metadata_map(path: str | Path, track_id_col: str = "track_id") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get(track_id_col, "")).strip()
            if tid:
                out[tid] = {str(k): str(v) for k, v in row.items()}
    return out


def _resolve_audio_paths(
    source_audio: str | None,
    style_audio: str | None,
    metadata: str | None,
    source_track: str | None,
    style_track: str | None,
    track_id_col: str,
    audio_col: str,
) -> tuple[str, str, dict[str, Any]]:
    source_audio = source_audio.strip() if source_audio else None
    style_audio = style_audio.strip() if style_audio else None
    if source_audio and style_audio:
        return source_audio, style_audio, {"source": "direct_audio_paths"}

    if not metadata or not source_track or not style_track:
        raise ValueError("provide either (--source_audio and --style_audio) or (--metadata + --source_track + --style_track)")

    mp = _read_metadata_map(metadata, track_id_col=track_id_col)
    if source_track not in mp:
        raise ValueError(f"source track_id not found in metadata: {source_track}")
    if style_track not in mp:
        raise ValueError(f"style track_id not found in metadata: {style_track}")

    src_audio = str(mp[source_track].get(audio_col, "")).strip()
    sty_audio = str(mp[style_track].get(audio_col, "")).strip()
    if not src_audio:
        raise ValueError(f"source track missing audio path in column '{audio_col}'")
    if not sty_audio:
        raise ValueError(f"style track missing audio path in column '{audio_col}'")
    return src_audio, sty_audio, {"source": "metadata_track_id", "source_track_id": source_track, "style_track_id": style_track}


def main() -> None:
    ap = argparse.ArgumentParser(description="Waveform-level style transfer (spectral statistics transfer).")
    ap.add_argument("--out_wav", required=True, help="output wav path")

    ap.add_argument("--source_audio", default=None, help="source wav/mp3 path")
    ap.add_argument("--style_audio", default=None, help="style wav/mp3 path")

    ap.add_argument("--metadata", default=None, help="metadata csv containing track_id and audio_path")
    ap.add_argument("--source_track", default=None, help="source track_id in metadata")
    ap.add_argument("--style_track", default=None, help="style track_id in metadata")
    ap.add_argument("--track_id_col", default="track_id")
    ap.add_argument("--audio_col", default="audio_path")

    ap.add_argument("--alpha", type=float, default=0.7, help="style intensity")
    ap.add_argument("--target_sr", type=int, default=24000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--max_seconds", type=float, default=12.0)
    ap.add_argument("--peak_norm", type=float, default=0.98)
    ap.add_argument("--report_json", default=None)
    args = ap.parse_args()

    src_audio, sty_audio, meta = _resolve_audio_paths(
        source_audio=args.source_audio,
        style_audio=args.style_audio,
        metadata=args.metadata,
        source_track=args.source_track,
        style_track=args.style_track,
        track_id_col=str(args.track_id_col),
        audio_col=str(args.audio_col),
    )

    out = transfer_waveform_style(
        source_audio_path=src_audio,
        style_audio_path=sty_audio,
        output_wav_path=args.out_wav,
        alpha=float(args.alpha),
        target_sr=int(args.target_sr),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        win_length=int(args.win_length),
        max_seconds=float(args.max_seconds) if args.max_seconds and float(args.max_seconds) > 0 else None,
        peak_norm=float(args.peak_norm),
    )

    payload = {
        "output_path": out.output_path,
        "sample_rate": out.sample_rate,
        "n_samples": out.n_samples,
        "source_audio_path": out.source_audio_path,
        "style_audio_path": out.style_audio_path,
        "metrics": out.metrics,
        "meta": meta,
    }
    if args.report_json:
        p = Path(str(args.report_json))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

