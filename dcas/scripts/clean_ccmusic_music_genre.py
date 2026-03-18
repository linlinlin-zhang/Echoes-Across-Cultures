from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ZH_LANGUAGE_ALIASES = {
    "zh",
    "zh-cn",
    "zh-hans",
    "zh-hant",
    "zh-tw",
    "zh-hk",
    "cmn",
    "yue",
    "zho",
    "chinese",
    "mandarin",
    "cantonese",
    "putonghua",
    "中文",
    "汉语",
    "漢語",
    "华语",
    "華語",
    "国语",
    "國語",
    "普通话",
    "普通話",
    "粤语",
    "粵語",
}

NON_ZH_LANGUAGE_ALIASES = {
    "en",
    "eng",
    "english",
    "ja",
    "jpn",
    "japanese",
    "jp",
    "ko",
    "kor",
    "korean",
    "fr",
    "fra",
    "french",
    "de",
    "deu",
    "german",
    "es",
    "spa",
    "spanish",
    "ru",
    "rus",
    "russian",
    "it",
    "ita",
    "italian",
    "pt",
    "por",
    "portuguese",
    "ar",
    "ara",
    "arabic",
    "hi",
    "hin",
    "hindi",
    "tr",
    "tur",
    "turkish",
}

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")
TRAILING_DURATION_RE = re.compile(r"\s*,?\s*\d+(?:\.\d+)?\s*s\s*$", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    text = text.replace("\ufeff", "")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = SPACE_RE.sub(" ", text)
    return text.strip(" \t\r\n\"'")


def _normalize_key(value: str) -> str:
    text = _normalize_text(value).casefold()
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1")
    last_error: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                rows = list(reader)
                fields = list(reader.fieldnames or [])
                return rows, fields, enc
        except UnicodeDecodeError as e:
            last_error = e
    raise RuntimeError(f"failed to decode csv: {path}") from last_error


def _expand_inputs(inputs: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(x for x in p.glob("*.csv") if x.is_file()))
        elif p.is_file():
            out.append(p)
        else:
            matches = sorted(Path().glob(str(raw)))
            out.extend(x for x in matches if x.is_file())
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _normalize_language(value: Any) -> tuple[str, str]:
    text = _normalize_text(value)
    if text == "":
        return "", "unknown"
    lowered = text.casefold()
    lowered = lowered.replace("_", "-")
    lowered = SPACE_RE.sub(" ", lowered)
    if lowered in ZH_LANGUAGE_ALIASES:
        return "zh", "zh_explicit"
    if lowered in NON_ZH_LANGUAGE_ALIASES:
        return lowered, "non_zh_explicit"
    if "chinese" in lowered or "mandarin" in lowered or "cantonese" in lowered:
        return "zh", "zh_explicit"
    if any(token in lowered for token in ("中文", "汉语", "漢語", "华语", "華語", "国语", "國語", "普通话", "普通話", "粤语", "粵語")):
        return "zh", "zh_explicit"
    if any(token in lowered for token in ("english", "japanese", "korean", "french", "german", "spanish", "russian", "hindi", "turkish")):
        return lowered, "non_zh_explicit"
    return lowered, "unknown"


def _text_signal(value: str) -> str:
    text = _normalize_text(value)
    if text == "":
        return "empty"
    has_han = bool(HAN_RE.search(text))
    has_kana = bool(KANA_RE.search(text))
    has_hangul = bool(HANGUL_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    if has_kana or has_hangul:
        return "non_zh_cjk"
    if has_han and not has_latin:
        return "han_only"
    if has_han and has_latin:
        return "mixed_han_latin"
    if has_latin:
        return "latin_only"
    return "other"


def _derive_singer_and_title(
    row: dict[str, Any],
    singer_col: str,
    title_col: str,
    language_col: str,
    file_name_col: str,
) -> tuple[str, str, str]:
    singer = _normalize_text(row.get(singer_col))
    title = _normalize_text(row.get(title_col))
    file_name = _normalize_text(row.get(file_name_col))
    if file_name != "":
        file_name = TRAILING_DURATION_RE.sub("", file_name).strip()
    if singer == "" and file_name and " - " in file_name:
        left, right = file_name.split(" - ", 1)
        if left.strip() and right.strip():
            singer = _normalize_text(left)
            if title == "":
                title = _normalize_text(right)
    if title == "" and file_name:
        if singer and file_name.casefold().startswith(f"{singer.casefold()} - "):
            title = _normalize_text(file_name[len(singer) + 3 :])
        elif " - " in file_name:
            left, right = file_name.split(" - ", 1)
            if singer == "" or _normalize_key(left) == _normalize_key(singer):
                title = _normalize_text(right)
            else:
                title = file_name
        else:
            title = file_name
    language_raw = _normalize_text(row.get(language_col))
    return singer, title, language_raw


def _decide_candidate(language_signal: str, singer_signal: str, title_signal: str) -> tuple[str, str]:
    if language_signal == "non_zh_explicit":
        return "drop", "explicit_non_zh_language"
    if singer_signal == "non_zh_cjk" or title_signal == "non_zh_cjk":
        if language_signal == "zh_explicit":
            return "review", "language_zh_but_text_contains_non_zh_cjk"
        return "drop", "non_zh_cjk_text"
    if language_signal == "zh_explicit":
        if singer_signal in {"han_only", "mixed_han_latin"} or title_signal in {"han_only", "mixed_han_latin"}:
            return "keep", "explicit_zh_language_with_han_text"
        return "review", "explicit_zh_language_but_text_ambiguous"
    if singer_signal == "han_only" and title_signal == "han_only":
        return "keep", "han_singer_and_han_title"
    if singer_signal in {"han_only", "mixed_han_latin"} and title_signal in {"han_only", "mixed_han_latin", "empty", "other"}:
        return "review", "partial_han_evidence_without_language"
    if title_signal in {"han_only", "mixed_han_latin"} and singer_signal in {"han_only", "mixed_han_latin", "empty", "other"}:
        return "review", "partial_han_evidence_without_language"
    if singer_signal == "latin_only" and title_signal == "latin_only":
        return "drop", "latin_only_without_zh_evidence"
    if singer_signal == "empty" and title_signal == "empty":
        return "drop", "missing_singer_and_title"
    return "drop", "insufficient_zh_evidence"


def clean_ccmusic_music_genre(
    inputs: list[str | Path],
    out_dir: str | Path,
    singer_col: str = "singer",
    language_col: str = "language",
    title_col: str = "title",
    file_name_col: str = "file_name",
) -> dict[str, Any]:
    input_paths = _expand_inputs(inputs)
    if not input_paths:
        raise RuntimeError("no input csv files found")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    all_fields: list[str] = []
    seen_fields: set[str] = set()
    sources: list[dict[str, Any]] = []
    seen_dedup: dict[str, str] = {}
    duplicates = 0

    for path in input_paths:
        rows, fields, encoding = _read_csv_rows(path)
        for field in fields:
            if field not in seen_fields:
                seen_fields.add(field)
                all_fields.append(field)
        kept_local = 0
        review_local = 0
        drop_local = 0
        dup_local = 0
        for idx, row in enumerate(rows):
            singer, title, language_raw = _derive_singer_and_title(
                row=row,
                singer_col=singer_col,
                title_col=title_col,
                language_col=language_col,
                file_name_col=file_name_col,
            )
            language_norm, language_signal = _normalize_language(language_raw)
            singer_signal = _text_signal(singer)
            title_signal = _text_signal(title)
            decision, reason = _decide_candidate(
                language_signal=language_signal,
                singer_signal=singer_signal,
                title_signal=title_signal,
            )

            dedup_key = "||".join(
                [
                    _normalize_key(singer),
                    _normalize_key(language_norm or language_raw or "unknown"),
                    _normalize_key(title),
                ]
            )
            duplicate_of = ""
            is_duplicate = "false"
            if dedup_key != "||||":
                previous = seen_dedup.get(dedup_key)
                if previous is not None:
                    duplicate_of = previous
                    is_duplicate = "true"
                    decision = "drop"
                    reason = "duplicate_singer_language_title"
                    duplicates += 1
                    dup_local += 1
                else:
                    seen_dedup[dedup_key] = f"{path.name}:{idx + 2}"

            row_out = dict(row)
            row_out["clean_source_file"] = str(path.resolve())
            row_out["clean_source_row"] = str(idx + 2)
            row_out["clean_singer"] = singer
            row_out["clean_title"] = title
            row_out["clean_language_raw"] = language_raw
            row_out["clean_language_norm"] = language_norm
            row_out["clean_language_signal"] = language_signal
            row_out["clean_singer_signal"] = singer_signal
            row_out["clean_title_signal"] = title_signal
            row_out["clean_decision"] = decision
            row_out["clean_reason"] = reason
            row_out["clean_dedup_key"] = dedup_key
            row_out["clean_is_duplicate"] = is_duplicate
            row_out["clean_duplicate_of"] = duplicate_of
            all_rows.append({k: _normalize_text(v) for k, v in row_out.items()})

            if decision == "keep":
                kept_local += 1
            elif decision == "review":
                review_local += 1
            else:
                drop_local += 1

        sources.append(
            {
                "path": str(path.resolve()),
                "encoding": encoding,
                "rows": int(len(rows)),
                "keep": int(kept_local),
                "review": int(review_local),
                "drop": int(drop_local),
                "duplicates": int(dup_local),
            }
        )

    added_fields = [
        "clean_source_file",
        "clean_source_row",
        "clean_singer",
        "clean_title",
        "clean_language_raw",
        "clean_language_norm",
        "clean_language_signal",
        "clean_singer_signal",
        "clean_title_signal",
        "clean_decision",
        "clean_reason",
        "clean_dedup_key",
        "clean_is_duplicate",
        "clean_duplicate_of",
    ]
    cols = [c for c in all_fields if c not in added_fields] + added_fields

    def _write_csv(name: str, rows: list[dict[str, str]]) -> Path:
        out_path = out_root / name
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in cols})
        return out_path

    keep_rows = [r for r in all_rows if r.get("clean_decision") == "keep"]
    review_rows = [r for r in all_rows if r.get("clean_decision") == "review"]
    drop_rows = [r for r in all_rows if r.get("clean_decision") == "drop"]

    all_csv = _write_csv("all_scored.csv", all_rows)
    keep_csv = _write_csv("keep.csv", keep_rows)
    review_csv = _write_csv("review.csv", review_rows)
    drop_csv = _write_csv("drop.csv", drop_rows)

    decision_counter = Counter(str(r.get("clean_decision", "")).strip() for r in all_rows)
    reason_counter = Counter(str(r.get("clean_reason", "")).strip() for r in all_rows)
    signal_counter = Counter(str(r.get("clean_language_signal", "")).strip() for r in all_rows)

    report = {
        "inputs": [str(p.resolve()) for p in input_paths],
        "out_dir": str(out_root.resolve()),
        "summary": {
            "rows_total": int(len(all_rows)),
            "rows_keep": int(len(keep_rows)),
            "rows_review": int(len(review_rows)),
            "rows_drop": int(len(drop_rows)),
            "rows_duplicate_drop": int(duplicates),
        },
        "decision_distribution": dict(sorted(decision_counter.items())),
        "reason_distribution": dict(sorted(reason_counter.items())),
        "language_signal_distribution": dict(sorted(signal_counter.items())),
        "sources": sources,
        "outputs": {
            "all_scored_csv": str(all_csv.resolve()),
            "keep_csv": str(keep_csv.resolve()),
            "review_csv": str(review_csv.resolve()),
            "drop_csv": str(drop_csv.resolve()),
        },
        "assumptions": [
            "Interprets strict secondary cleaning as retaining only high-confidence Chinese candidates.",
            "Uses the compound key singer + language + title after normalization for exact deduplication.",
            "If language is missing, only rows with strong Han-character evidence in singer/title can survive strict filtering.",
        ],
    }
    report_path = out_root / "clean_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Strict secondary cleaner for the CCMusic Music Genre metadata using singer/language/title."
    )
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV files or directories containing CCMusic music-genre metadata")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--singer_col", default="singer")
    ap.add_argument("--language_col", default="language")
    ap.add_argument("--title_col", default="title")
    ap.add_argument("--file_name_col", default="file_name")
    args = ap.parse_args()

    out = clean_ccmusic_music_genre(
        inputs=[str(x) for x in args.inputs],
        out_dir=str(args.out_dir),
        singer_col=str(args.singer_col),
        language_col=str(args.language_col),
        title_col=str(args.title_col),
        file_name_col=str(args.file_name_col),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
