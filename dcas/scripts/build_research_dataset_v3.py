from __future__ import annotations

import argparse
import ast
import csv
import html
import io
import json
import math
import numpy as np
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import reverse_geocoder as rg
import torchaudio
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcas.scripts.merge_metadata_dedup import merge_metadata_dedup

DEFAULT_OUT_ROOT = ROOT / "storage" / "public" / "research_dataset_v3"
DEFAULT_RAW_ROOT = DEFAULT_OUT_ROOT / "raw_sources"
DEFAULT_CACHE_ROOT = DEFAULT_OUT_ROOT / "_cache"

FMA_REPO_ID = "benjamin-paine/free-music-archive-full"
FMA_METADATA_ZIP = ROOT / "tmp" / "fma_metadata.zip"
OPENCPOP_SONGLIST_URL = "https://wenet-e2e.github.io/opencpop/resources/songlist/"
CHINA_JINGJU_TARGET = 30
CHINA_OPENCPOP_TARGET = 50

COUNTRY_PATTERNS: dict[str, list[str]] = {
    "great_britain": [
        "united kingdom",
        "great britain",
        "england",
        "scotland",
        "wales",
        "london",
        "manchester",
        "liverpool",
        "birmingham",
        "bristol",
        "leeds",
        "sheffield",
        "cardiff",
        "glasgow",
        "edinburgh",
        "nottingham",
        "newcastle",
        "brighton",
        "oxford",
        "cambridge",
        "belfast",
        "uk",
    ],
    "france": [
        "france",
        "paris",
        "lyon",
        "marseille",
        "toulouse",
        "bordeaux",
        "lille",
        "nantes",
        "rennes",
        "strasbourg",
        "montpellier",
        "grenoble",
        "nice",
    ],
    "germany": [
        "germany",
        "berlin",
        "hamburg",
        "munich",
        "muenchen",
        "munchen",
        "cologne",
        "koln",
        "frankfurt",
        "leipzig",
        "dresden",
        "stuttgart",
        "dortmund",
        "bremen",
        "hannover",
    ],
    "italy": [
        "italy",
        "roma",
        "rome",
        "milano",
        "milan",
        "torino",
        "turin",
        "napoli",
        "naples",
        "bologna",
        "firenze",
        "florence",
        "venezia",
        "venice",
        "palermo",
        "genoa",
        "genova",
    ],
    "russia": [
        "russia",
        "russian federation",
        "moscow",
        "moskva",
        "saint petersburg",
        "st petersburg",
        "sankt petersburg",
        "novosibirsk",
        "yekaterinburg",
        "ekaterinburg",
        "kazan",
        "nizhny novgorod",
        "samara",
    ],
}

FMA_COUNTRY_CODES = {
    "great_britain": "GB",
    "france": "FR",
    "germany": "DE",
    "italy": "IT",
    "russia": "RU",
}

FMA_SUPPLEMENT_COUNTRY_PATTERNS: dict[str, list[str]] = {
    "indonesia": [
        "indonesia",
        "jakarta",
        "bandung",
        "yogyakarta",
        "surabaya",
        "denpasar",
        "bali",
        "java",
        "sumatra",
        "sulawesi",
    ]
}

FMA_SUPPLEMENT_COUNTRY_CODES = {
    "indonesia": "ID",
}

FMA_BANNED_GENRES = {
    "experimental",
    "electronic",
    "rock",
    "novelty",
    "international",
    "spoken",
    "hip-hop",
}

ANGLO_POP_BANNED_TERMS = [
    "experimental",
    "electronic",
    "rock",
    "novelty",
    "international",
    "spoken",
    "hip-hop",
    "hiphop",
]


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "item"


def _norm_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\u0400-\u04ff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_audio_exts(filename: str) -> str:
    stem = filename
    while True:
        new_stem, ext = os.path.splitext(stem)
        if ext.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".au"}:
            stem = new_stem
            continue
        break
    return stem


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _safe_console_text(value: Any) -> str:
    text = _to_text(value)
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _clean_optional_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _contains_normalized_term(text: Any, term: str) -> bool:
    hay = _norm_text(text)
    needle = _norm_text(term)
    if hay == "" or needle == "":
        return False
    return re.search(rf"(^|\s){re.escape(needle)}($|\s)", hay) is not None


def _duration_from_bytes(data: bytes) -> float:
    info = torchaudio.info(io.BytesIO(data))
    return float(info.num_frames / float(info.sample_rate))


def _duration_from_file(path: str | Path) -> float:
    info = torchaudio.info(str(path))
    return float(info.num_frames / float(info.sample_rate))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    preferred = [
        "track_id",
        "culture",
        "audio_path",
        "source_dataset",
        "source_split",
        "source_index",
        "label",
        "substyle",
        "instrument",
        "language",
        "title",
        "artist",
        "duration_sec",
        "license",
        "license_note",
        "region",
        "instrument_family",
        "era",
        "notes",
        "url",
    ]
    for col in preferred:
        for row in rows:
            if col in row and col not in seen:
                seen.add(col)
                fields.append(col)
                break
    for row in rows:
        for col in row.keys():
            if col not in seen:
                seen.add(col)
                fields.append(col)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _to_text(row.get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _copy_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)


def _copy_file(src: str | Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        while True:
            chunk = rf.read(1024 * 1024)
            if not chunk:
                break
            wf.write(chunk)


def _round_robin_diverse(
    rows: list[dict[str, Any]],
    target_n: int,
    artist_key: str,
    max_per_artist: int = 3,
) -> list[dict[str, Any]]:
    grouped: dict[str, deque[dict[str, Any]]] = {}
    for row in rows:
        artist = _norm_text(row.get(artist_key) or "")
        if artist == "":
            artist = "__unknown__"
        grouped.setdefault(artist, deque()).append(row)

    selected: list[dict[str, Any]] = []
    taken = Counter()
    artist_order = sorted(grouped.keys(), key=lambda k: (-len(grouped[k]), k))

    while len(selected) < target_n:
        progress = False
        for artist in artist_order:
            if len(selected) >= target_n:
                break
            bucket = grouped[artist]
            if not bucket or taken[artist] >= max_per_artist:
                continue
            selected.append(bucket.popleft())
            taken[artist] += 1
            progress = True
        if not progress:
            break
    if len(selected) < target_n:
        leftovers: list[dict[str, Any]] = []
        for artist in artist_order:
            leftovers.extend(list(grouped[artist]))
        selected.extend(leftovers[: max(0, target_n - len(selected))])
    return selected


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = sorted(_safe_float(row.get("duration_sec")) for row in rows)
    durations = [x for x in durations if x is not None]
    artists = {_norm_text(row.get("artist", "")) for row in rows if _norm_text(row.get("artist", ""))}
    return {
        "n_rows": len(rows),
        "n_artists": len(artists),
        "duration_min": min(durations) if durations else None,
        "duration_median": durations[len(durations) // 2] if durations else None,
        "duration_max": max(durations) if durations else None,
        "sources": sorted({str(row.get("source_dataset", "")) for row in rows}),
    }


def _domain_out(out_root: Path, culture: str) -> Path:
    path = out_root / culture
    (path / "audio").mkdir(parents=True, exist_ok=True)
    return path


def _build_india(out_root: Path, raw_root: Path) -> Path:
    out_dir = _domain_out(out_root, "india")
    metadata_path = out_dir / "metadata.csv"
    if metadata_path.exists():
        return metadata_path

    zip_path = raw_root / "saraga_hindustani.zip"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        audio_names = sorted(
            n
            for n in zf.namelist()
            if not n.startswith("__MACOSX/")
            and os.path.splitext(n)[1].lower() in {".mp3", ".wav", ".flac"}
        )
        for idx, name in enumerate(audio_names):
            payload = zf.read(name)
            duration = _duration_from_bytes(payload)
            if duration < 30:
                continue
            rel = name.split("/", 1)[1]
            parts = rel.split("/")
            album_block = parts[0]
            title = _strip_audio_exts(Path(parts[-1]).name)
            piece = parts[-2] if len(parts) >= 3 else title
            artist = album_block.split(" by ", 1)[1].strip() if " by " in album_block else ""
            track_id = f"india_saraga_{idx:04d}"
            ext = Path(name).suffix.lower() or ".mp3"
            dst = out_dir / "audio" / f"{track_id}{ext}"
            _copy_bytes(dst, payload)
            rows.append(
                {
                    "track_id": track_id,
                    "culture": "india",
                    "audio_path": str(Path("audio") / dst.name),
                    "source_dataset": "saraga_hindustani",
                    "source_split": "full_zip",
                    "source_index": idx,
                    "label": piece,
                    "substyle": "hindustani_art_music",
                    "title": title,
                    "artist": artist,
                    "duration_sec": round(duration, 6),
                    "license": "cc-by-nc-sa-4.0",
                    "license_note": "Saraga Hindustani audio released under CC BY-NC-SA 4.0",
                    "region": "india",
                    "era": "traditional",
                    "notes": album_block,
                }
            )
    _write_csv(metadata_path, rows)
    return metadata_path


def _build_turkey(out_root: Path, target_n: int) -> Path:
    out_dir = _domain_out(out_root, "turkey")
    metadata_path = out_dir / "metadata.csv"
    if metadata_path.exists():
        return metadata_path

    src_csv = ROOT / "storage" / "public" / "research_dataset_v2" / "turkey" / "metadata.csv"
    src_rows = _read_csv(src_csv)
    kept: list[dict[str, Any]] = []
    for row in src_rows:
        src_audio = Path(row["audio_path"])
        if not src_audio.is_absolute():
            src_audio = (src_csv.parent / src_audio).resolve()
        duration = _duration_from_file(src_audio)
        if duration < 30:
            continue
        kept.append(
            {
                "src_audio": src_audio,
                "duration_sec": round(duration, 6),
                **row,
            }
        )
    selected = kept[:target_n]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        track_id = f"turkey_modern_{idx:04d}"
        ext = Path(str(row["src_audio"])).suffix.lower() or ".mp3"
        dst = out_dir / "audio" / f"{track_id}{ext}"
        _copy_file(row["src_audio"], dst)
        rows.append(
            {
                "track_id": track_id,
                "culture": "turkey",
                "audio_path": str(Path("audio") / dst.name),
                "source_dataset": row.get("source_dataset", "bilal63/turkish_music_emotion_dataset"),
                "source_split": row.get("source_split", "train"),
                "source_index": row.get("source_index", idx),
                "label": row.get("label", ""),
                "substyle": "modern_turkish_song",
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "duration_sec": row["duration_sec"],
                "region": "turkey",
                "era": "modern",
                "notes": "OTMM Makam archive available locally only as pitch/json; v3 main uses modern Turkish audio source.",
            }
        )
    _write_csv(metadata_path, rows)
    return metadata_path


def _build_jingju_rows(out_dir: Path, raw_root: Path) -> list[dict[str, Any]]:
    zip_path = raw_root / "jingju_acappella_wav.zip"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        audio_names = sorted(
            n
            for n in zf.namelist()
            if not n.startswith("__MACOSX/") and os.path.splitext(n)[1].lower() == ".wav"
        )
        out_idx = 0
        for src_idx, name in enumerate(audio_names):
            payload = zf.read(name)
            duration = _duration_from_bytes(payload)
            if duration < 30:
                continue
            stem = _strip_audio_exts(Path(name).name)
            parts = stem.split("-")
            role = parts[0] if parts else ""
            excerpt = parts[1] if len(parts) > 1 else stem
            work = parts[2] if len(parts) > 2 else ""
            suffix = parts[3] if len(parts) > 3 else ""
            track_id = f"china_jingju_{out_idx:04d}"
            out_idx += 1
            dst = out_dir / "audio" / f"{track_id}.wav"
            _copy_bytes(dst, payload)
            rows.append(
                {
                    "track_id": track_id,
                    "culture": "china",
                    "audio_path": str(Path("audio") / dst.name),
                    "source_dataset": "compmusic_jingju_acappella",
                    "source_split": "wav_zip",
                    "source_index": src_idx,
                    "label": role,
                    "substyle": "jingju_acappella",
                    "language": "zh",
                    "title": excerpt,
                    "artist": suffix,
                    "duration_sec": round(duration, 6),
                    "license": "cc-by-nc-4.0_mixed",
                    "license_note": "UPF/lon recordings CC BY-NC 4.0; qm recordings follow Isophonics license note.",
                    "region": "china",
                    "instrument_family": "voice",
                    "era": "traditional",
                    "notes": work,
                }
            )
    return rows


def _uniform_subsample_rows(rows: list[dict[str, Any]], target_n: int | None) -> list[dict[str, Any]]:
    if target_n is None or int(target_n) <= 0 or len(rows) <= int(target_n):
        return list(rows)
    idx = np.linspace(0, len(rows) - 1, num=int(target_n), dtype=int).tolist()
    return [rows[int(i)] for i in idx]


def _build_ctis_rows(limit: int | None = None) -> list[dict[str, Any]]:
    src_csv = ROOT / "storage" / "public" / "research_dataset_v2" / "china" / "metadata.csv"
    src_rows = _read_csv(src_csv)
    rows: list[dict[str, Any]] = []
    out_idx = 0
    for row in src_rows:
        src_audio = Path(row["audio_path"])
        if not src_audio.is_absolute():
            src_audio = (src_csv.parent / src_audio).resolve()
        duration = _duration_from_file(src_audio)
        if duration < 30:
            continue
        rows.append(
            {
                "src_audio": src_audio,
                "track_id": f"china_ctis_{out_idx:04d}",
                "culture": "china",
                "source_dataset": row.get("source_dataset", "ccmusic-database/CTIS"),
                "source_split": row.get("source_split", "train"),
                "source_index": row.get("source_index", out_idx),
                "label": row.get("label", ""),
                "substyle": "traditional_instrumental",
                "instrument": row.get("cname", "") or row.get("pinyin", ""),
                "title": row.get("cname", "") or row.get("pinyin", "") or row.get("label", ""),
                "artist": "",
                "duration_sec": round(duration, 6),
                "license_note": "CTIS source-level licensing should be checked before redistribution.",
                "region": "china",
                "instrument_family": "traditional_instrument",
                "era": "traditional",
                "notes": row.get("pinyin", ""),
            }
        )
        out_idx += 1
    if limit is not None:
        rows = rows[:limit]
    return rows


def _load_opencpop_songlist(raw_root: Path) -> pd.DataFrame:
    opencpop_root = raw_root / "opencpop"
    opencpop_root.mkdir(parents=True, exist_ok=True)
    cache_csv = opencpop_root / "songlist.csv"
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    df = pd.read_html(OPENCPOP_SONGLIST_URL)[0]
    df = df.rename(
        columns={
            "Song Id": "song_id",
            "Song Name": "song_name",
            "Time Signature": "time_signature",
            "BPM": "bpm",
        }
    )
    df["song_id"] = df["song_id"].astype(int)
    df["bpm"] = df["bpm"].astype(float)
    df.to_csv(cache_csv, index=False, encoding="utf-8")
    return df


def _select_opencpop_songlist(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    if len(df) <= int(target_n):
        return df.sort_values("song_id").reset_index(drop=True)
    ranked = df.sort_values(["bpm", "song_id"]).reset_index(drop=True)
    idx = np.linspace(0, len(ranked) - 1, num=int(target_n), dtype=int)
    picked = ranked.iloc[idx].drop_duplicates(subset=["song_id"]).sort_values("song_id").reset_index(drop=True)
    return picked


def _build_opencpop_rows(
    out_dir: Path,
    raw_root: Path,
    password: str | None,
    target_n: int,
) -> list[dict[str, Any]]:
    if int(target_n) <= 0:
        return []
    zip_path = raw_root / "opencpop" / "wavs_raw.zip"
    if not zip_path.exists():
        alt = raw_root / "opencpop" / "wavs.zip"
        if alt.exists():
            zip_path = alt
    if not zip_path.exists():
        raise FileNotFoundError(f"OpenCpop wav zip not found: {zip_path}")
    if password is None or str(password).strip() == "":
        raise RuntimeError("OpenCpop password is required. Set OPENCPOP_ZIP_PASSWORD or pass --opencpop_password.")

    songlist = _load_opencpop_songlist(raw_root)
    chosen = _select_opencpop_songlist(songlist, target_n=int(target_n))

    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for out_idx, song in enumerate(chosen.to_dict(orient="records")):
            song_id = int(song["song_id"])
            zip_name = f"{2000 + song_id}.wav"
            payload = zf.read(zip_name, pwd=str(password).encode("utf-8"))
            duration = _duration_from_bytes(payload)
            if duration < 30:
                continue
            track_id = f"china_opencpop_{out_idx:04d}"
            dst = out_dir / "audio" / f"{track_id}.wav"
            _copy_bytes(dst, payload)
            rows.append(
                {
                    "track_id": track_id,
                    "culture": "china",
                    "audio_path": str(Path("audio") / dst.name),
                    "source_dataset": "OpenCpop",
                    "source_split": "wavs_raw",
                    "source_index": song_id,
                    "label": "mandarin_pop",
                    "substyle": "mandarin_pop_singing",
                    "instrument": "voice",
                    "language": "zh",
                    "title": str(song["song_name"]),
                    "artist": "opencpop_single_singer",
                    "duration_sec": round(duration, 6),
                    "license": "cc-by-nc-nd-4.0",
                    "license_note": "OpenCpop raw song-level wavs; authorized local access, non-commercial usage only.",
                    "region": "china",
                    "instrument_family": "voice",
                    "era": "modern",
                    "notes": f"song_id={song_id}; bpm={float(song['bpm'])}; time_signature={song['time_signature']}; single-singer Mandarin pop corpus",
                }
            )
    return rows


def _build_china(
    out_root: Path,
    raw_root: Path,
    jingju_target: int = CHINA_JINGJU_TARGET,
    opencpop_target: int = CHINA_OPENCPOP_TARGET,
    opencpop_password: str | None = None,
) -> Path:
    out_dir = _domain_out(out_root, "china")
    metadata_path = out_dir / "metadata.csv"
    if metadata_path.exists():
        return metadata_path

    rows = _uniform_subsample_rows(_build_jingju_rows(out_dir, raw_root), target_n=int(jingju_target))
    ctis_rows = _build_ctis_rows()
    for row in ctis_rows:
        dst = out_dir / "audio" / f"{row['track_id']}{Path(str(row['src_audio'])).suffix.lower() or '.wav'}"
        _copy_file(row["src_audio"], dst)
        row["audio_path"] = str(Path("audio") / dst.name)
        row.pop("src_audio", None)
        rows.append(row)
    rows.extend(
        _build_opencpop_rows(
            out_dir=out_dir,
            raw_root=raw_root,
            password=opencpop_password,
            target_n=int(opencpop_target),
        )
    )
    _write_csv(metadata_path, rows)
    return metadata_path


def _parse_list_label(value: str) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).lower() for x in parsed]
    except Exception:
        pass
    return [str(value).lower()]


def _build_anglo_pop(out_root: Path, target_n: int) -> Path:
    out_dir = _domain_out(out_root, "modern_english_pop")
    metadata_path = out_dir / "metadata.csv"
    if metadata_path.exists():
        return metadata_path

    src_csv = ROOT / "storage" / "public" / "research_dataset_v2" / "anglo_pop" / "metadata.csv"
    src_rows = _read_csv(src_csv)
    filtered: list[dict[str, Any]] = []
    for row in src_rows:
        labels = _parse_list_label(row.get("label", ""))
        joined = " ".join(labels)
        if not any("pop" in x for x in labels):
            continue
        if any(term in joined for term in ANGLO_POP_BANNED_TERMS):
            continue
        src_audio = Path(row["audio_path"])
        if not src_audio.is_absolute():
            src_audio = (src_csv.parent / src_audio).resolve()
        duration = _duration_from_file(src_audio)
        if duration < 30:
            continue
        filtered.append({"src_audio": src_audio, "duration_sec": round(duration, 6), **row})
    selected = filtered[:target_n]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        track_id = f"modern_english_pop_{idx:04d}"
        ext = Path(str(row["src_audio"])).suffix.lower() or ".mp3"
        dst = out_dir / "audio" / f"{track_id}{ext}"
        _copy_file(row["src_audio"], dst)
        rows.append(
            {
                "track_id": track_id,
                "culture": "modern_english_pop",
                "audio_path": str(Path("audio") / dst.name),
                "source_dataset": row.get("source_dataset", "vtsouval/mtg_jamendo_autotagging"),
                "source_split": row.get("source_split", "train"),
                "source_index": row.get("source_index", idx),
                "label": row.get("label", ""),
                "substyle": "modern_pop_benchmark",
                "instrument": row.get("instrument", ""),
                "title": "",
                "artist": "",
                "duration_sec": row["duration_sec"],
                "region": "anglophone",
                "era": "modern",
                "notes": "Proxy anglophone pop benchmark from MTG-Jamendo pop-tagged tracks with excluded genres removed.",
            }
        )
    _write_csv(metadata_path, rows)
    return metadata_path


def _build_indonesia_probe(out_root: Path, raw_root: Path) -> Path:
    out_dir = _domain_out(out_root, "indonesia_probe")
    metadata_path = out_dir / "metadata.csv"
    if metadata_path.exists():
        return metadata_path

    zip_path = raw_root / "gamelan_music_dataset.zip"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        audio_names = sorted(
            n
            for n in zf.namelist()
            if not n.startswith("__MACOSX/") and os.path.splitext(n)[1].lower() == ".wav"
        )
        out_idx = 0
        for src_idx, name in enumerate(audio_names):
            payload = zf.read(name)
            duration = _duration_from_bytes(payload)
            if duration < 30:
                continue
            rel = name.split("/", 1)[1]
            category = "/".join(rel.split("/")[1:3]) if len(rel.split("/")) >= 3 else ""
            if not category.startswith("orchestra/"):
                continue
            track_id = f"indonesia_probe_{out_idx:04d}"
            out_idx += 1
            dst = out_dir / "audio" / f"{track_id}.wav"
            _copy_bytes(dst, payload)
            rows.append(
                {
                    "track_id": track_id,
                    "culture": "indonesia_probe",
                    "audio_path": str(Path("audio") / dst.name),
                    "source_dataset": "gamelan_music_dataset",
                    "source_split": "zip_orchestra",
                    "source_index": src_idx,
                    "label": category,
                    "substyle": "gamelan_orchestra",
                    "duration_sec": round(duration, 6),
                    "license": "cc-by-4.0",
                    "region": "indonesia",
                    "era": "traditional",
                    "notes": rel,
                }
            )
    _write_csv(metadata_path, rows)
    return metadata_path


def _build_fma_indonesia_targets(cache_root: Path) -> list[dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "fma_indonesia_targets.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached:
            return cached

    tracks, genres = _load_fma_tracks_and_genres()
    parent = {int(row["genre_id"]): int(row["parent"]) for _, row in genres.iterrows()}
    genre_name = {int(row["genre_id"]): str(row["title"]).strip().lower() for _, row in genres.iterrows()}
    genre_title = {int(row["genre_id"]): str(row["title"]).strip() for _, row in genres.iterrows()}
    lineage_cache: dict[int, set[str]] = {}

    def row_is_banned(genres_all: Any) -> bool:
        if pd.isna(genres_all):
            return False
        try:
            ids = ast.literal_eval(str(genres_all))
        except Exception:
            return False
        for gid in ids:
            gid = int(gid)
            if gid not in lineage_cache:
                cur = gid
                lineage: set[str] = set()
                seen: set[int] = set()
                while cur and cur not in seen:
                    seen.add(cur)
                    lineage.add(genre_name.get(cur, ""))
                    cur = parent.get(cur, 0)
                lineage_cache[gid] = lineage
            if lineage_cache[gid] & FMA_BANNED_GENRES:
                return True
        return False

    coords: list[tuple[float, float]] = []
    coord_indices: list[int] = []
    for idx, row in tracks.iterrows():
        lat = _safe_float(row.get("artist__latitude"))
        lon = _safe_float(row.get("artist__longitude"))
        if lat is None or lon is None:
            continue
        coords.append((lat, lon))
        coord_indices.append(int(idx))

    reverse_country: dict[int, str] = {}
    if coords:
        results = rg.search(coords, mode=1)
        for idx, result in zip(coord_indices, results):
            cc = str(result.get("cc", "")).upper()
            if cc == FMA_SUPPLEMENT_COUNTRY_CODES["indonesia"]:
                reverse_country[idx] = "indonesia"

    def map_text_country(location: Any) -> str | None:
        for pattern in FMA_SUPPLEMENT_COUNTRY_PATTERNS["indonesia"]:
            if _contains_normalized_term(location, pattern):
                return "indonesia"
        return None

    items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for idx, row in tracks.iterrows():
        if row_is_banned(row.get("track__genres_all")):
            continue
        duration = _parse_fma_duration(row.get("track__duration"))
        if duration is None or duration < 30:
            continue
        culture = reverse_country.get(int(idx))
        match_source = "geo"
        if culture is None:
            culture = map_text_country(row.get("artist__location"))
            match_source = "text"
        if culture != "indonesia":
            continue
        page_url = _clean_optional_text(row.get("track_url"))
        if not page_url.startswith("http"):
            continue
        genres_all = _clean_optional_text(row.get("track__genres_all"))
        genre_titles: list[str] = []
        if genres_all:
            try:
                for gid in ast.literal_eval(genres_all):
                    gid_int = int(gid)
                    title = genre_title.get(gid_int)
                    if title:
                        genre_titles.append(title)
            except Exception:
                genre_titles = []
        item = {
            "culture": "indonesia",
            "artist": str(row.get("artist__name") or ""),
            "title": str(row.get("track__title") or ""),
            "album_title": str(row.get("album__title") or ""),
            "artist_location": str(row.get("artist__location") or ""),
            "duration_sec": round(duration, 6),
            "license": str(row.get("track__license") or ""),
            "language": _clean_optional_text(row.get("track__language_code")),
            "genres_all": genres_all,
            "genre_titles": genre_titles,
            "page_url": page_url,
            "track_listens": _safe_float(row.get("track__listens")) or 0.0,
            "track_favorites": _safe_float(row.get("track__favorites")) or 0.0,
            "match_key": "||".join(
                [
                    _norm_text(row.get("artist__name") or ""),
                    _norm_text(row.get("track__title") or ""),
                    _norm_text(row.get("album__title") or ""),
                ]
            ),
            "match_source": match_source,
        }
        dedup_key = item["page_url"] or item["match_key"]
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        items.append(item)

    items = sorted(
        items,
        key=lambda row: (-float(row["track_favorites"]), -float(row["track_listens"]), row["match_key"]),
    )
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return items


def _download_fma_supplement_rows(
    items: list[dict[str, Any]],
    out_dir: Path,
    culture: str,
    track_prefix: str,
    substyle: str,
    era: str,
    workers: int,
) -> list[dict[str, Any]]:
    base_headers = {"User-Agent": "Mozilla/5.0 (Codex dataset builder)"}

    def prepare(item: dict[str, Any]) -> dict[str, Any]:
        session = requests.Session()
        session.headers.update(base_headers)
        file_url = _fetch_fma_file_url(session, item["page_url"])
        ext = Path(urlparse(file_url).path).suffix.lower() or ".mp3"
        return {**item, "file_url": file_url, "ext": ext}

    prepared: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        futures = {ex.submit(prepare, item): item for item in items}
        for i, fut in enumerate(as_completed(futures), start=1):
            item = futures[fut]
            try:
                prepared.append(fut.result())
            except Exception as exc:
                print(
                    f"[fma-page] {culture}: skip "
                    f"{_safe_console_text(item.get('artist'))} - {_safe_console_text(item.get('title'))} "
                    f"({_safe_console_text(exc)})"
                )
            if i % 10 == 0 or i == len(items):
                print(f"[fma-page] {culture}: resolved {i}/{len(items)} file URLs")

    prepared = sorted(prepared, key=lambda row: row["match_key"])

    def download_one(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        session = requests.Session()
        session.headers.update(base_headers)
        track_id = f"{track_prefix}_{idx:04d}"
        dst = out_dir / "audio" / f"{track_id}{item['ext']}"
        _download_file(session, item["file_url"], dst)
        duration = _duration_from_file(dst)
        label_value = json.dumps(item.get("genre_titles", []), ensure_ascii=False)
        return {
            "track_id": track_id,
            "culture": culture,
            "audio_path": str(Path("audio") / dst.name),
            "source_dataset": "Free Music Archive",
            "source_split": "country_filtered_nonwestern",
            "source_index": idx,
            "label": label_value,
            "substyle": substyle,
            "title": item.get("title", ""),
            "artist": item.get("artist", ""),
            "language": item.get("language", ""),
            "duration_sec": round(duration, 6),
            "license": item.get("license", ""),
            "license_note": "Direct fileUrl scraped from public FMA track page.",
            "region": culture,
            "era": era,
            "notes": item.get("artist_location", ""),
            "url": item.get("page_url", ""),
        }

    rows_out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        futures = {ex.submit(download_one, idx, item): item for idx, item in enumerate(prepared)}
        for i, fut in enumerate(as_completed(futures), start=1):
            item = futures[fut]
            try:
                rows_out.append(fut.result())
            except Exception as exc:
                print(
                    f"[fma-download] {culture}: skip "
                    f"{_safe_console_text(item.get('artist'))} - {_safe_console_text(item.get('title'))} "
                    f"({_safe_console_text(exc)})"
                )
            if i % 10 == 0 or i == len(prepared):
                print(f"[fma-download] {culture}: downloaded {i}/{len(prepared)} attempts")
    return sorted(rows_out, key=lambda row: row["track_id"])


def _build_indonesia(out_root: Path, raw_root: Path, cache_root: Path, workers: int) -> Path:
    out_dir = _domain_out(out_root, "indonesia")
    metadata_path = out_dir / "metadata.csv"
    if metadata_path.exists():
        return metadata_path

    rows: list[dict[str, Any]] = []
    zip_path = raw_root / "gamelan_music_dataset.zip"
    with zipfile.ZipFile(zip_path) as zf:
        audio_names = sorted(
            n
            for n in zf.namelist()
            if not n.startswith("__MACOSX/") and os.path.splitext(n)[1].lower() == ".wav"
        )
        out_idx = 0
        for src_idx, name in enumerate(audio_names):
            payload = zf.read(name)
            duration = _duration_from_bytes(payload)
            if duration < 30:
                continue
            rel = name.split("/", 1)[1]
            category = "/".join(rel.split("/")[1:3]) if len(rel.split("/")) >= 3 else ""
            if not category.startswith("orchestra/"):
                continue
            track_id = f"indonesia_gamelan_{out_idx:04d}"
            out_idx += 1
            dst = out_dir / "audio" / f"{track_id}.wav"
            _copy_bytes(dst, payload)
            rows.append(
                {
                    "track_id": track_id,
                    "culture": "indonesia",
                    "audio_path": str(Path("audio") / dst.name),
                    "source_dataset": "gamelan_music_dataset",
                    "source_split": "zip_orchestra",
                    "source_index": src_idx,
                    "label": category,
                    "substyle": "gamelan_orchestra",
                    "duration_sec": round(duration, 6),
                    "license": "cc-by-4.0",
                    "region": "indonesia",
                    "era": "traditional",
                    "notes": rel,
                }
            )

    fma_rows = _download_fma_supplement_rows(
        items=_build_fma_indonesia_targets(cache_root),
        out_dir=out_dir,
        culture="indonesia",
        track_prefix="indonesia_fma",
        substyle="modern_indonesian_supplement",
        era="modern",
        workers=workers,
    )
    rows.extend(fma_rows)
    rows = sorted(rows, key=lambda row: row["track_id"])
    _write_csv(metadata_path, rows)
    return metadata_path


def _parse_fma_duration(value: Any) -> float | None:
    text = str(value or "").strip()
    if text == "" or text == "nan":
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(text)
    except Exception:
        return None


def _load_fma_tracks_and_genres() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(FMA_METADATA_ZIP) as zf:
        tracks = pd.read_csv(zf.open("fma_metadata/tracks.csv"), header=[0, 1], low_memory=False)
        raw_tracks = pd.read_csv(zf.open("fma_metadata/raw_tracks.csv"), low_memory=False)
        genres = pd.read_csv(zf.open("fma_metadata/genres.csv"))
    tracks = tracks.iloc[1:].copy()
    tracks.columns = [
        f"{a}__{b}" if not str(b).startswith("Unnamed") else str(a)
        for a, b in tracks.columns.to_list()
    ]
    tracks["track_id"] = pd.to_numeric(tracks["Unnamed: 0_level_0"], errors="coerce").astype("Int64")
    raw_subset = raw_tracks[["track_id", "track_url", "track_file", "album_url", "artist_url"]].copy()
    raw_subset["track_id"] = pd.to_numeric(raw_subset["track_id"], errors="coerce").astype("Int64")
    tracks = tracks.merge(raw_subset, on="track_id", how="left")
    return tracks, genres


def _build_fma_selected_targets(cache_root: Path, per_country: int, strict_min: bool = True) -> list[dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    mode_tag = "strict" if strict_min else "flex"
    cache_path = cache_root / f"fma_selected_targets_{per_country}_{mode_tag}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached and all(str(item.get("page_url") or "").strip().startswith("http") for item in cached):
            return cached

    tracks, genres = _load_fma_tracks_and_genres()
    parent = {int(row["genre_id"]): int(row["parent"]) for _, row in genres.iterrows()}
    genre_name = {int(row["genre_id"]): str(row["title"]).strip().lower() for _, row in genres.iterrows()}
    lineage_cache: dict[int, set[str]] = {}

    def row_is_banned(genres_all: Any) -> bool:
        if pd.isna(genres_all):
            return False
        try:
            ids = ast.literal_eval(str(genres_all))
        except Exception:
            return False
        for gid in ids:
            gid = int(gid)
            if gid not in lineage_cache:
                cur = gid
                lineage: set[str] = set()
                seen: set[int] = set()
                while cur and cur not in seen:
                    seen.add(cur)
                    lineage.add(genre_name.get(cur, ""))
                    cur = parent.get(cur, 0)
                lineage_cache[gid] = lineage
            if lineage_cache[gid] & FMA_BANNED_GENRES:
                return True
        return False

    coords: list[tuple[float, float]] = []
    coord_indices: list[int] = []
    for idx, row in tracks.iterrows():
        lat = _safe_float(row.get("artist__latitude"))
        lon = _safe_float(row.get("artist__longitude"))
        if lat is None or lon is None:
            continue
        coords.append((lat, lon))
        coord_indices.append(int(idx))

    reverse_country: dict[int, str] = {}
    if coords:
        results = rg.search(coords, mode=1)
        for idx, result in zip(coord_indices, results):
            cc = str(result.get("cc", "")).upper()
            for culture, short in FMA_COUNTRY_CODES.items():
                if cc == short:
                    reverse_country[idx] = culture
                    break

    def map_text_country(location: Any) -> str | None:
        loc = _norm_text(location)
        if loc == "":
            return None
        for culture, patterns in COUNTRY_PATTERNS.items():
            for pattern in patterns:
                if _norm_text(pattern) in loc:
                    return culture
        return None

    candidates_by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in tracks.iterrows():
        if row_is_banned(row.get("track__genres_all")):
            continue
        duration = _parse_fma_duration(row.get("track__duration"))
        if duration is None or duration < 30:
            continue
        culture = reverse_country.get(int(idx))
        if culture is None:
            culture = map_text_country(row.get("artist__location"))
        if culture not in FMA_COUNTRY_CODES:
            continue
        candidates_by_country[culture].append(
            {
                "culture": culture,
                "artist": str(row.get("artist__name") or ""),
                "title": str(row.get("track__title") or ""),
                "album_title": str(row.get("album__title") or ""),
                "artist_location": str(row.get("artist__location") or ""),
                "duration_sec": round(duration, 6),
                "license": str(row.get("track__license") or ""),
                "language": str(row.get("track__language_code") or ""),
                "track_listens": _safe_float(row.get("track__listens")) or 0.0,
                "track_favorites": _safe_float(row.get("track__favorites")) or 0.0,
                "genres_all": str(row.get("track__genres_all") or ""),
                "page_url": _clean_optional_text(row.get("track_url")),
                "source_track_file": _clean_optional_text(row.get("track_file")),
                "artist_key": _norm_text(row.get("artist__name") or ""),
                "match_key": "||".join(
                    [
                        _norm_text(row.get("artist__name") or ""),
                        _norm_text(row.get("track__title") or ""),
                        _norm_text(row.get("album__title") or ""),
                    ]
                ),
                "match_key_loose": "||".join(
                    [
                        _norm_text(row.get("artist__name") or ""),
                        _norm_text(row.get("track__title") or ""),
                    ]
                ),
            }
        )

    selected: list[dict[str, Any]] = []
    for culture in sorted(FMA_COUNTRY_CODES.keys()):
        items = sorted(
            candidates_by_country[culture],
            key=lambda row: (-float(row["track_favorites"]), -float(row["track_listens"]), row["match_key"]),
        )
        wanted = min(per_country, len(items)) if not strict_min else per_country
        picked = _round_robin_diverse(items, wanted, artist_key="artist_key", max_per_artist=3)
        if strict_min and len(picked) < per_country:
            raise RuntimeError(
                f"FMA country '{culture}' only produced {len(picked)} rows after diversity selection (< {per_country})"
            )
        selected.extend(picked)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    return selected


def _resolve_fma_page_urls(selected: list[dict[str, Any]], cache_root: Path) -> list[dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "fma_selected_with_urls.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if (
            len(cached) == len(selected)
            and cached
            and all(str(item.get("page_url") or "").strip().startswith("http") for item in cached)
        ):
            return cached

    updated = [dict(item) for item in selected if str(item.get("page_url") or "").strip().startswith("http")]
    missing = len(selected) - len(updated)
    if missing > 0:
        print(f"[fma-meta] skipped {missing} targets with missing page_url in local metadata")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    return updated


def _fetch_fma_file_url(session: requests.Session, page_url: str) -> str:
    resp = session.get(page_url, timeout=60)
    resp.raise_for_status()
    match = re.search(r"data-track-info='([^']+)'", resp.text)
    if not match:
        raise RuntimeError(f"missing data-track-info on {page_url}")
    raw = html.unescape(match.group(1))
    payload = json.loads(raw)
    file_url = str(payload.get("fileUrl") or "").strip()
    if file_url == "":
        raise RuntimeError(f"missing fileUrl on {page_url}")
    return file_url


def _download_file(session: requests.Session, url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)


def _download_fma_rows(
    rows: list[dict[str, Any]],
    out_root: Path,
    workers: int,
    target_per_culture: int,
) -> dict[str, Path]:
    base_headers = {"User-Agent": "Mozilla/5.0 (Codex dataset builder)"}
    metadata_paths: dict[str, Path] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["culture"]].append(row)

    for culture, items in grouped.items():
        out_dir = _domain_out(out_root, culture)
        metadata_path = out_dir / "metadata.csv"
        if metadata_path.exists():
            metadata_paths[culture] = metadata_path
            continue

        ranked_items = [{**item, "source_rank": idx} for idx, item in enumerate(items)]

        def prepare(item: dict[str, Any]) -> dict[str, Any]:
            session = requests.Session()
            session.headers.update(base_headers)
            file_url = _fetch_fma_file_url(session, item["page_url"])
            ext = Path(urlparse(file_url).path).suffix.lower() or ".mp3"
            return {**item, "file_url": file_url, "ext": ext}

        prepared: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
            futures = {ex.submit(prepare, item): item for item in ranked_items}
            for i, fut in enumerate(as_completed(futures), start=1):
                item = futures[fut]
                try:
                    prepared.append(fut.result())
                except Exception as exc:
                    print(
                        f"[fma-page] {culture}: skip "
                        f"{_safe_console_text(item.get('artist'))} - {_safe_console_text(item.get('title'))} "
                        f"({_safe_console_text(exc)})"
                    )
                if i % 25 == 0 or i == len(items):
                    print(f"[fma-page] {culture}: resolved {i}/{len(items)} file URLs")

        prepared = sorted(prepared, key=lambda row: int(row["source_rank"]))
        if len(prepared) < target_per_culture:
            raise RuntimeError(
                f"FMA country '{culture}' only resolved {len(prepared)} file URLs (< {target_per_culture})"
            )

        def download_one(idx: int, item: dict[str, Any]) -> dict[str, Any]:
            session = requests.Session()
            session.headers.update(base_headers)
            track_id = f"{culture}_{idx:04d}"
            dst = out_dir / "audio" / f"{track_id}{item['ext']}"
            _download_file(session, item["file_url"], dst)
            duration = _duration_from_file(dst)
            return {
                "track_id": track_id,
                "culture": culture,
                "audio_path": str(Path("audio") / dst.name),
                "source_dataset": "Free Music Archive",
                "source_split": "country_filtered",
                "source_index": idx,
                "label": item.get("genres_all", ""),
                "language": item.get("language", ""),
                "title": item.get("title", ""),
                "artist": item.get("artist", ""),
                "duration_sec": round(duration, 6),
                "license": item.get("license", ""),
                "license_note": "Direct fileUrl scraped from public FMA track page.",
                "region": culture,
                "era": "mixed",
                "notes": item.get("artist_location", ""),
                "url": item.get("page_url", ""),
            }

        rows_out: list[dict[str, Any]] = []
        batch_size = max(4, workers * 3)
        next_rank = 0
        attempt_n = 0
        while len(rows_out) < target_per_culture and next_rank < len(prepared):
            batch = prepared[next_rank : next_rank + batch_size]
            next_rank += len(batch)
            with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
                futures = {
                    ex.submit(download_one, attempt_n + offset, item): item
                    for offset, item in enumerate(batch)
                }
                for i, fut in enumerate(as_completed(futures), start=1):
                    item = futures[fut]
                    try:
                        rows_out.append(fut.result())
                    except Exception as exc:
                        print(
                            f"[fma-download] {culture}: skip "
                            f"{_safe_console_text(item.get('artist'))} - {_safe_console_text(item.get('title'))} "
                            f"({_safe_console_text(exc)})"
                        )
                    if (len(rows_out) % 25 == 0 and len(rows_out) > 0) or len(rows_out) >= target_per_culture:
                        print(f"[fma-download] {culture}: downloaded {len(rows_out)}/{target_per_culture}")
            attempt_n += len(batch)

        if len(rows_out) < target_per_culture:
            raise RuntimeError(
                f"FMA country '{culture}' only downloaded {len(rows_out)} rows (< {target_per_culture})"
            )

        selected_rows = sorted(rows_out, key=lambda row: row["source_index"])[:target_per_culture]
        keep_paths: set[Path] = set()
        for idx, row in enumerate(selected_rows):
            old_rel = Path(str(row["audio_path"]))
            old_abs = out_dir / old_rel
            ext = old_abs.suffix.lower() or ".mp3"
            new_name = f"{culture}_{idx:04d}{ext}"
            new_abs = out_dir / "audio" / new_name
            new_abs.parent.mkdir(parents=True, exist_ok=True)
            if old_abs.exists() and old_abs != new_abs:
                if new_abs.exists():
                    new_abs.unlink()
                old_abs.replace(new_abs)
            keep_paths.add(new_abs)
            row["track_id"] = f"{culture}_{idx:04d}"
            row["source_index"] = idx
            row["audio_path"] = str(Path("audio") / new_name)
        audio_dir = out_dir / "audio"
        if audio_dir.exists():
            for path in audio_dir.iterdir():
                if path.is_file() and path not in keep_paths:
                    path.unlink()
        rows_out = sorted(selected_rows, key=lambda row: row["track_id"])
        _write_csv(metadata_path, rows_out)
        metadata_paths[culture] = metadata_path
    return metadata_paths


def _build_fma_western(out_root: Path, cache_root: Path, per_country: int, workers: int) -> dict[str, Path]:
    reserve_per_country = per_country + max(200, per_country * 2)
    selected = _build_fma_selected_targets(cache_root, per_country=reserve_per_country, strict_min=False)
    with_urls = _resolve_fma_page_urls(selected, cache_root)
    return _download_fma_rows(with_urls, out_root, workers=workers, target_per_culture=per_country)


def _write_summary(out_root: Path, metadata_paths: list[Path], main_merge_path: Path) -> Path:
    summary = {
        "out_root": str(out_root.resolve()),
        "main_metadata_csv": str(main_merge_path.resolve()),
        "domains": [],
    }
    for path in metadata_paths:
        rows = _read_csv(path)
        culture = rows[0]["culture"] if rows else path.parent.name
        summary["domains"].append(
            {
                "culture": culture,
                "metadata_csv": str(path.resolve()),
                **_summarize_rows(rows),
            }
        )
    summary["domains"] = sorted(summary["domains"], key=lambda row: row["culture"])
    summary_path = out_root / "summary_v3_main.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def build_research_dataset_v3(
    out_root: Path,
    raw_root: Path,
    cache_root: Path,
    fma_per_country: int,
    turkey_target: int,
    anglo_pop_target: int,
    workers: int,
    china_jingju_target: int = CHINA_JINGJU_TARGET,
    china_opencpop_target: int = CHINA_OPENCPOP_TARGET,
    opencpop_password: str | None = None,
) -> dict[str, Any]:
    random.seed(0)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    india = _build_india(out_root, raw_root)
    turkey = _build_turkey(out_root, target_n=turkey_target)
    china = _build_china(
        out_root,
        raw_root,
        jingju_target=int(china_jingju_target),
        opencpop_target=int(china_opencpop_target),
        opencpop_password=opencpop_password,
    )
    indonesia = _build_indonesia(out_root, raw_root, cache_root, workers=workers)
    anglo = _build_anglo_pop(out_root, target_n=anglo_pop_target)
    indonesia_probe = _build_indonesia_probe(out_root, raw_root)
    western = _build_fma_western(out_root, cache_root, per_country=fma_per_country, workers=workers)

    main_paths = [
        india,
        turkey,
        china,
        indonesia,
        anglo,
        western["germany"],
        western["france"],
        western["italy"],
        western["great_britain"],
        western["russia"],
    ]
    merge_path = out_root / "metadata_v3_main.csv"
    merge_report = merge_metadata_dedup(inputs=main_paths, out_csv=merge_path)
    summary_path = _write_summary(out_root, main_paths, merge_path)
    return {
        "main_metadata": str(merge_path.resolve()),
        "merge_report": merge_report,
        "summary_json": str(summary_path.resolve()),
        "indonesia_metadata": str(indonesia.resolve()),
        "indonesia_probe_metadata": str(indonesia_probe.resolve()),
        "domains": [str(path.resolve()) for path in main_paths],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build research_dataset_v3 from locally audited sources and FMA country filters.")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--raw_root", default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--fma_per_country", type=int, default=110)
    ap.add_argument("--turkey_target", type=int, default=150)
    ap.add_argument("--anglo_pop_target", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--china_jingju_target", type=int, default=CHINA_JINGJU_TARGET)
    ap.add_argument("--china_opencpop_target", type=int, default=CHINA_OPENCPOP_TARGET)
    ap.add_argument("--opencpop_password", default=None)
    args = ap.parse_args()

    out = build_research_dataset_v3(
        out_root=Path(args.out_root),
        raw_root=Path(args.raw_root),
        cache_root=Path(args.cache_root),
        fma_per_country=int(args.fma_per_country),
        turkey_target=int(args.turkey_target),
        anglo_pop_target=int(args.anglo_pop_target),
        workers=int(args.workers),
        china_jingju_target=int(args.china_jingju_target),
        china_opencpop_target=int(args.china_opencpop_target),
        opencpop_password=args.opencpop_password or os.environ.get("OPENCPOP_ZIP_PASSWORD"),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
