"""Package only public website/catalog assets, never credentials, uploads or model weights."""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/exhibition-edge.tar.gz")
    args = parser.parse_args()
    files = [
        (p, p.relative_to(ROOT).as_posix())
        for p in (ROOT / "web").rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and not any(
            part in {"screenshots", "node_modules", ".git", "__pycache__"} for part in p.relative_to(ROOT / "web").parts
        )
        and p.suffix.lower()
        in {
            ".html",
            ".js",
            ".css",
            ".json",
            ".svg",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".ico",
            ".woff",
            ".woff2",
            ".txt",
        }
    ]
    for name in [
        "__init__.py",
        "paths.py",
        "catalog_search.py",
        "description_translations.py",
        "artist_locations.py",
        "track_locations.py",
        "lightweight_catalog.py",
    ]:
        files.append((ROOT / "dcas_server" / name, "dcas_server/" + name))
    for name in [
        "metadata_merged.csv",
        "artist_locations.json",
        "descriptions_en.json",
        "catalog_market_locations.json",
    ]:
        files.append((ROOT / "storage/public/merged" / name, "storage/public/merged/" + name))
    files.append((ROOT / "deploy/exhibition-edge/edge_app.py", "edge_app.py"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz", compresslevel=4) as archive:
        for file, name in sorted(files):
            assert not any(
                part in {"configs", "secrets", "uploads"} or part.startswith(".") for part in Path(name).parts
            )
            archive.add(file, arcname=name, recursive=False)
    print(f"{args.output}: {len(files)} public files, {args.output.stat().st_size} bytes")
    print("SHA256 " + hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
