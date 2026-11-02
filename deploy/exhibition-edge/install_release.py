"""Install a public bundle on the existing cloud VM, validating all audio mappings first."""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == args.sha256, "Archive checksum mismatch"
    base = Path("/srv/echo-exhibition")
    release = base / "releases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release.mkdir(parents=True)
    with tarfile.open(args.archive) as archive:
        assert all(not m.issym() and not m.islnk() for m in archive.getmembers()), "Links are not allowed"
        archive.extractall(release, filter="data")
    media = Path("/srv/echo/cloud_audio_10gb")
    rows = list(
        csv.DictReader((release / "storage/public/merged/metadata_merged.csv").open(encoding="utf-8-sig", newline=""))
    )
    all_files = {p.stem: p for p in sorted(media.glob("*/audio/**/*.m4a"))}
    index = {}
    for row in rows:
        relative = row["audio_path"].replace("\\", "/").split("/storage/public/", 1)[-1]
        file = (media / relative).with_suffix(".m4a")
        if not file.is_file():
            file = all_files.get(row["track_id"])
        if file and file.is_file() and file.resolve().is_relative_to(media):
            index[row["track_id"]] = file.relative_to(media).as_posix()
    assert len(index) == len(rows) == 30000, "Not all catalog tracks have cloud audio"
    (release / "audio-index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    for file in release.rglob("*"):
        file.chmod(0o755 if file.is_dir() else 0o644)
    current = base / "current"
    previous = current.resolve() if current.exists() else None

    def activate(target):
        temporary = base / "next"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, current)
        subprocess.run(["systemctl", "restart", "echo-exhibition"], check=True)

    try:
        activate(release)
        for _ in range(20):
            try:
                with urllib.request.urlopen("http://127.0.0.1:18030/api/edge/health", timeout=2) as stream:
                    result = json.load(stream)
                if result["catalog_tracks"] == result["cloud_audio_tracks"] == 30000:
                    print(json.dumps({"release": str(release), "previous": str(previous), "health": result}))
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError("Cloud catalog did not become ready")
    except Exception:
        if previous:
            activate(previous)
        raise


if __name__ == "__main__":
    main()
