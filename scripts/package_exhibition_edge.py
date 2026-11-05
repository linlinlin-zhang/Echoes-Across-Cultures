"""Package only public website/catalog assets, never credentials, uploads or model weights."""

from __future__ import annotations
import argparse
import hashlib
import gzip
import io
import json
import re
import time
from pathlib import Path
import tarfile
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

ROOT = Path(__file__).resolve().parents[1]


def version_assets(content: bytes, page: Path, assets: dict[str, str]) -> bytes:
    """Pin local asset references to their content, including fetch('...json')."""

    def replace(match):
        quote, url = match.group(1), match.group(2)
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            return match.group(0)
        target = (ROOT / "web" / parts.path.lstrip("/")) if parts.path.startswith("/") else page.parent / parts.path
        try:
            relative = target.resolve().relative_to((ROOT / "web").resolve()).as_posix()
        except ValueError:
            return match.group(0)
        digest = assets.get(relative)
        if not digest:
            return match.group(0)
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "v"]
        query.append(("v", digest))
        return (
            quote
            + urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    parts.path,
                    urlencode(query),
                    parts.fragment,
                )
            )
            + quote
        )

    pattern = r"""(["'])([^"'\s<>]+\.(?:js|css|json|svg|png|jpe?g|webp|woff2?)(?:\?[^"'\s<>]*)?(?:\#[^"'\s<>]*)?)\1"""
    return re.sub(pattern, replace, content.decode("utf-8")).encode("utf-8")


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
        "genre_metadata.py",
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
    assets = {
        name.removeprefix("web/"): hashlib.sha256(file.read_bytes()).hexdigest()[:12]
        for file, name in files
        if name.startswith("web/") and file.suffix.lower() != ".html"
    }
    compressed_count = 0
    compressed_bytes_saved = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz", compresslevel=4) as archive:
        for file, name in sorted(files):
            assert not any(
                part in {"configs", "secrets", "uploads"} or part.startswith(".") for part in Path(name).parts
            )
            if name.startswith("web/"):
                content = file.read_bytes()
                if file.suffix.lower() == ".html":
                    content = version_assets(content, file, assets)
                info = archive.gettarinfo(file, arcname=name)
                info.size = len(content)
                if file.suffix.lower() == ".html":
                    info.mtime = int(time.time())
                archive.addfile(info, io.BytesIO(content))
                if file.suffix.lower() in {".html", ".js", ".css", ".json", ".svg"} and len(content) >= 1024:
                    compressed = gzip.compress(content, compresslevel=9, mtime=0)
                    gzinfo = archive.gettarinfo(file, arcname=name + ".gz")
                    gzinfo.size = len(compressed)
                    gzinfo.mtime = info.mtime
                    archive.addfile(gzinfo, io.BytesIO(compressed))
                    compressed_count += 1
                    compressed_bytes_saved += len(content) - len(compressed)
            else:
                archive.add(file, arcname=name, recursive=False)
        manifest = json.dumps(
            {
                "assets": assets,
                "precompressed_files": compressed_count,
                "bytes_saved_per_full_asset_transfer": compressed_bytes_saved,
            },
            indent=2,
        ).encode()
        info = tarfile.TarInfo("public-asset-manifest.json")
        info.size = len(manifest)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest))
    print(f"{args.output}: {len(files)} public files, {args.output.stat().st_size} bytes")
    print("SHA256 " + hashlib.sha256(args.output.read_bytes()).hexdigest())
    print(f"Precompressed {compressed_count} assets; {compressed_bytes_saved} fewer bytes per full asset transfer")


if __name__ == "__main__":
    main()
