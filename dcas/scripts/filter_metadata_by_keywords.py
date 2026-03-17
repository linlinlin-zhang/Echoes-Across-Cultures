from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def _parse_cell(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if text == "":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [text]


def _matches_keywords(values: list[str], include: list[str], exclude: list[str]) -> bool:
    lowered = [v.lower() for v in values]
    has_include = True if not include else any(any(k in v for k in include) for v in lowered)
    has_exclude = any(any(k in v for k in exclude) for v in lowered)
    return has_include and not has_exclude


def filter_metadata_by_keywords(
    in_csv: str | Path,
    out_dir: str | Path,
    columns: list[str],
    include_keywords: list[str],
    exclude_keywords: list[str] | None = None,
    max_rows: int | None = None,
    copy_audio: bool = True,
) -> dict[str, Any]:
    in_path = Path(in_csv)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_audio = out_root / "audio"
    if copy_audio:
        out_audio.mkdir(parents=True, exist_ok=True)

    exclude = [x.lower().strip() for x in (exclude_keywords or []) if x.strip()]
    include = [x.lower().strip() for x in include_keywords if x.strip()]

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    kept: list[dict[str, str]] = []
    copied = 0
    for row in rows:
        values: list[str] = []
        for col in columns:
            values.extend(_parse_cell(row.get(col)))
        if not _matches_keywords(values=values, include=include, exclude=exclude):
            continue
        rr = dict(row)
        src_audio = Path(str(row.get("audio_path", "")).strip())
        if src_audio == Path(""):
            continue
        if not src_audio.is_absolute():
            src_audio = (in_path.parent / src_audio).resolve()
        if not src_audio.exists():
            continue
        if copy_audio:
            dst_audio = out_audio / src_audio.name
            if not dst_audio.exists():
                shutil.copyfile(src_audio, dst_audio)
                copied += 1
            rr["audio_path"] = str(Path("audio") / src_audio.name)
        else:
            rr["audio_path"] = str(src_audio)
        kept.append(rr)
        if max_rows is not None and len(kept) >= int(max_rows):
            break

    out_csv = out_root / "metadata.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    report = {
        "input_csv": str(in_path.resolve()),
        "output_csv": str(out_csv.resolve()),
        "columns": columns,
        "include_keywords": include,
        "exclude_keywords": exclude,
        "max_rows": None if max_rows is None else int(max_rows),
        "matched_rows": int(len(kept)),
        "copied_audio_files": int(copied),
        "copy_audio": bool(copy_audio),
    }
    report_path = out_root / "filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter metadata rows by keyword matches in one or more columns.")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--columns", required=True, help="Comma-separated metadata columns to inspect")
    ap.add_argument("--include", required=True, help="Comma-separated include keywords")
    ap.add_argument("--exclude", default="", help="Comma-separated exclude keywords")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--no_copy_audio", action="store_true")
    args = ap.parse_args()

    out = filter_metadata_by_keywords(
        in_csv=str(args.in_csv),
        out_dir=str(args.out_dir),
        columns=[x.strip() for x in str(args.columns).split(",") if x.strip()],
        include_keywords=[x.strip() for x in str(args.include).split(",") if x.strip()],
        exclude_keywords=[x.strip() for x in str(args.exclude).split(",") if x.strip()],
        max_rows=args.max_rows,
        copy_audio=not bool(args.no_copy_audio),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
