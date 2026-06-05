#!/usr/bin/env python3
"""Create a cloud-sized compressed copy of the iTunes and Jamendo audio pools."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".opus", ".webm"}
DEFAULT_FFMPEG = Path("/mnt/d/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe")
DATASETS = {
    "itunes_crawl": Path("storage/public/itunes_crawl"),
    "jamendo_crawl": Path("storage/public/jamendo_crawl"),
}


@lru_cache(maxsize=100000)
def to_windows_path_cached(path_text: str) -> str:
    path = Path(path_text)
    resolved = path.resolve()
    parts = resolved.parts
    if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        tail = "\\".join(parts[3:])
        return f"{drive}:\\{tail}" if tail else f"{drive}:\\"
    result = subprocess.run(
        ["wslpath", "-w", str(resolved)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return str(resolved)


def to_windows_path(path: Path) -> str:
    return to_windows_path_cached(str(path))


def iter_audio_files(root: Path, limit: int = 0) -> list[Path]:
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            files.append(path)
            if limit and len(files) >= limit:
                return files
    return files


def convert_one(task: tuple[Path, Path, Path, str, bool]) -> tuple[str, int, str]:
    src, dst, ffmpeg, bitrate, force = task
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return ("skipped", dst.stat().st_size, str(dst))

    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        to_windows_path(src),
        "-map_metadata",
        "0",
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        "-movflags",
        "+faststart",
        to_windows_path(tmp),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        message = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
        return ("failed", 0, f"{src}: {message[:600]}")

    os.replace(tmp, dst)
    return ("converted", dst.stat().st_size, str(dst))


def _metadata_rows(src_metadata: Path, *, audio_prefix: str = "") -> tuple[list[str], list[dict[str, str]]]:
    if not src_metadata.exists():
        return [], []
    with src_metadata.open("r", encoding="utf-8-sig", newline="") as src_file:
        reader = csv.DictReader(src_file)
        fieldnames = reader.fieldnames or []
        rows: list[dict[str, str]] = []
        for row in reader:
            audio_path = row.get("audio_path", "")
            if audio_path:
                stem = Path(audio_path).with_suffix(".m4a")
                rel_audio = str(stem).replace("\\", "/")
                row["audio_path"] = f"{audio_prefix.rstrip('/')}/{rel_audio}" if audio_prefix else rel_audio
            rows.append(row)
    return fieldnames, rows


def _write_metadata(dst_metadata: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    if not fieldnames:
        return
    dst_metadata.parent.mkdir(parents=True, exist_ok=True)
    with dst_metadata.open("w", encoding="utf-8", newline="") as dst_file:
        writer = csv.DictWriter(dst_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rewrite_metadata(src_metadata: Path, dst_metadata: Path) -> int:
    fieldnames, rows = _metadata_rows(src_metadata)
    _write_metadata(dst_metadata, fieldnames, rows)
    return len(rows)


def rewrite_merged_metadata(output_root: Path) -> int:
    fieldnames: list[str] = []
    field_seen: set[str] = set()
    merged_rows: list[dict[str, str]] = []

    for name, dataset_root in DATASETS.items():
        dataset_fields, rows = _metadata_rows(dataset_root / "metadata.csv", audio_prefix=name)
        for field in dataset_fields:
            if field not in field_seen:
                fieldnames.append(field)
                field_seen.add(field)
        merged_rows.extend(rows)

    _write_metadata(output_root / "metadata_merged.csv", fieldnames, merged_rows)
    return len(merged_rows)


def bytes_for(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def metadata_bytes(output_root: Path) -> int:
    return sum(path.stat().st_size for path in output_root.glob("*/metadata.csv") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="storage/public/cloud_audio_10gb")
    parser.add_argument("--bitrate", default="48k")
    parser.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--ffmpeg", default=str(DEFAULT_FFMPEG))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug only: convert at most N files")
    args = parser.parse_args()

    ffmpeg = Path(args.ffmpeg)
    if not ffmpeg.exists():
        print(f"ffmpeg not found: {ffmpeg}", file=sys.stderr)
        return 2

    output_root = Path(args.output_root)
    all_tasks: list[tuple[Path, Path, Path, str, bool]] = []
    planned = {}
    for name, dataset_root in DATASETS.items():
        audio_root = dataset_root / "audio"
        out_audio_root = output_root / name / "audio"
        files = iter_audio_files(audio_root, args.limit)
        planned[name] = {
            "source_audio_root": str(audio_root),
            "output_audio_root": str(out_audio_root),
            "files": len(files),
        }
        for src in files:
            rel = src.relative_to(audio_root).with_suffix(".m4a")
            all_tasks.append((src, out_audio_root / rel, ffmpeg, args.bitrate, args.force))
        rewrite_metadata(dataset_root / "metadata.csv", output_root / name / "metadata.csv")

    merged_metadata_rows = rewrite_merged_metadata(output_root)
    if args.metadata_only:
        print(f"metadata written: {merged_metadata_rows} rows in {output_root / 'metadata_merged.csv'}")
        return 0

    start = time.time()
    counters = {"converted": 0, "skipped": 0, "failed": 0}
    output_bytes = 0
    failures = []
    total = len(all_tasks)
    print(
        f"compressing {total} files to {output_root} at {args.bitrate} with {args.workers} workers",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(convert_one, task) for task in all_tasks]
        for done, future in enumerate(as_completed(futures), 1):
            status, size, detail = future.result()
            counters[status] += 1
            output_bytes += size
            if status == "failed":
                failures.append(detail)
            if done == total or done % 100 == 0:
                elapsed = max(0.1, time.time() - start)
                rate = done / elapsed
                print(
                    f"{done}/{total} converted={counters['converted']} skipped={counters['skipped']} "
                    f"failed={counters['failed']} output_gb={output_bytes / 1024**3:.2f} rate={rate:.2f}/s",
                    flush=True,
                )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_total": "about 10GB",
        "codec": "aac",
        "bitrate": args.bitrate,
        "output_root": str(output_root),
        "planned": planned,
        "merged_metadata_path": str(output_root / "metadata_merged.csv"),
        "merged_metadata_rows": merged_metadata_rows,
        "counters": counters,
        "output_bytes": output_bytes + metadata_bytes(output_root),
        "output_gb": round((output_bytes + metadata_bytes(output_root)) / 1024**3, 3),
        "failures": failures[:200],
        "failure_count": len(failures),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        print(
            f"completed with {len(failures)} failures; see {output_root / 'manifest.json'}",
            file=sys.stderr,
        )
        return 1
    print(f"done: {manifest['output_gb']} GiB in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
