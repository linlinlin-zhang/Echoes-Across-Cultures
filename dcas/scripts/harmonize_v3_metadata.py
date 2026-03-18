from __future__ import annotations

import argparse
import ast
import csv
import json
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FMA_METADATA_ZIP = REPO_ROOT / "tmp" / "fma_metadata.zip"


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_fma_genre_titles(zip_path: Path | None) -> dict[int, str]:
    if zip_path is None or not zip_path.exists():
        return {}
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("fma_metadata/genres.csv") as f:
            reader = csv.DictReader((line.decode("utf-8") for line in f))
            out: dict[int, str] = {}
            for row in reader:
                try:
                    out[int(row["genre_id"])] = str(row["title"]).strip()
                except Exception:
                    continue
            return out


def _parse_label_titles(raw_label: str, fma_genre_map: dict[int, str]) -> list[str]:
    text = str(raw_label or "").strip()
    if text == "":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            items = ast.literal_eval(text)
        except Exception:
            return []
        titles: list[str] = []
        for item in items:
            if isinstance(item, str):
                title = str(item).strip()
                if title:
                    titles.append(title)
            else:
                try:
                    mapped = fma_genre_map.get(int(item))
                except Exception:
                    mapped = None
                if mapped:
                    titles.append(str(mapped))
        return titles
    return [text]


def _norm(x: Any) -> str:
    return str(x or "").strip().lower()


def _coarse_label(row: dict[str, str], fma_genre_map: dict[int, str]) -> str:
    substyle = _norm(row.get("substyle"))
    source_dataset = _norm(row.get("source_dataset"))
    era = _norm(row.get("era"))
    instrument_family = _norm(row.get("instrument_family"))
    language = _norm(row.get("language"))
    genre_titles = [_norm(x) for x in _parse_label_titles(str(row.get("label") or ""), fma_genre_map)]
    text_blob = " | ".join(
        [
            substyle,
            _norm(row.get("instrument")),
            _norm(row.get("title")),
            _norm(row.get("artist")),
            " ".join(genre_titles),
        ]
    )

    if substyle == "jingju_acappella":
        return "traditional_vocal"
    if substyle == "traditional_instrumental":
        return "traditional_instrumental"
    if substyle == "hindustani_art_music":
        return "art_music"
    if substyle == "gamelan_orchestra":
        return "traditional_instrumental"
    if substyle == "mandarin_pop_singing":
        return "modern_pop_song"
    if substyle in {"modern_pop_benchmark", "modern_turkish_song", "modern_indonesian_supplement"}:
        return "modern_song"

    if source_dataset == "opencpop":
        return "modern_pop_song"
    if source_dataset == "saraga_hindustani":
        return "art_music"
    if source_dataset == "compmusic_jingju_acappella":
        return "traditional_vocal"
    if source_dataset == "ccmusic-database/ctis":
        return "traditional_instrumental"

    if any(x in text_blob for x in ["opera", "choral", "choir", "orchestra", "classical", "soundtrack"]):
        return "soundtrack_classical"
    if any(x in text_blob for x in ["jazz", "blues"]):
        return "jazz_blues"
    if any(x in text_blob for x in ["folk", "acoustic", "singer-songwriter", "singer songwriter", "chanson"]):
        return "folk_acoustic"
    if any(x in text_blob for x in ["ambient", "drone", "instrumental"]):
        return "instrumental_ambient"
    if any(x in text_blob for x in ["pop", "easy listening", "song"]):
        return "modern_song"

    if instrument_family == "voice" and era == "traditional":
        return "traditional_vocal"
    if instrument_family == "voice" and (era == "modern" or language in {"en", "zh", "de"}):
        return "modern_song"
    if instrument_family == "traditional_instrument":
        return "traditional_instrumental"
    if era == "traditional":
        return "traditional_instrumental"
    if era == "modern":
        return "modern_song"
    return "unknown"


def harmonize_metadata(
    metadata_csv: str | Path,
    out_csv: str | Path,
    fma_metadata_zip: str | Path | None = None,
) -> dict[str, Any]:
    in_path = Path(metadata_csv)
    out_path = Path(out_csv)
    rows, fieldnames = _read_rows(in_path)
    fma_genre_map = _load_fma_genre_titles(Path(fma_metadata_zip) if fma_metadata_zip else DEFAULT_FMA_METADATA_ZIP)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        language = str(item.get("language") or "").strip().lower()
        if language == "nan":
            language = ""
        item["language"] = language
        item["coarse_label"] = _coarse_label(item, fma_genre_map=fma_genre_map)
        item["is_instrumental"] = "1" if str(item.get("instrument_family") or "").strip().lower() == "traditional_instrument" else "0"
        enriched.append(item)

    final_fields = list(fieldnames)
    for extra in ["coarse_label", "is_instrumental"]:
        if extra not in final_fields:
            final_fields.append(extra)
    _write_rows(out_path, enriched, final_fields)

    counts: dict[str, int] = {}
    for row in enriched:
        key = str(row.get("coarse_label") or "unknown")
        counts[key] = counts.get(key, 0) + 1

    report = {
        "metadata": str(in_path.resolve()),
        "out": str(out_path.resolve()),
        "n_rows": int(len(enriched)),
        "coarse_label_counts": dict(sorted(counts.items())),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Add unified coarse metadata columns for V3.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fma_metadata_zip", default=str(DEFAULT_FMA_METADATA_ZIP))
    args = ap.parse_args()

    out = harmonize_metadata(
        metadata_csv=args.metadata,
        out_csv=args.out,
        fma_metadata_zip=args.fma_metadata_zip,
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
