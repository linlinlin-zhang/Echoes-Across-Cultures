"""
Self-checking supervisor for targeted iTunes culture crawl top-ups.

This wrapper repeatedly launches the resumable iTunes crawler until the
requested culture-level targets are met. It is meant for long unattended runs:
network/API failures are logged, checkpoints are reused, and the next attempt
continues from existing metadata/state.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from dcas.scripts.crawl_itunes_previews import COUNTRY_TO_CULTURE


DEFAULT_TARGETS = {
    "china": 1200,
    "japan": 1200,
    "korea": 1200,
    "india": 1200,
    "brazil": 1200,
    "latin": 1200,
    "africa": 1200,
    "middle_east": 1200,
    "southeast_asia": 1200,
    "celtic": 1200,
    "nordic": 1200,
    "eastern_europe": 1200,
    "balkans": 1200,
    "caribbean": 1200,
    "andean": 1199,
    "central_asia": 1199,
}

CUSTOM_CULTURE_COUNTRIES = {
    "celtic": ["IE", "GB"],
    "nordic": ["SE", "NO", "FI", "DK", "IS"],
    "eastern_europe": ["PL", "CZ", "HU", "RO", "SK", "UA"],
    "balkans": ["GR", "BG", "HR", "RS", "SI", "BA", "ME", "MK", "AL"],
    "caribbean": ["JM", "DO", "TT", "BB", "BS", "PR"],
    "andean": ["PE", "CO", "CL", "AR", "EC", "BO"],
    "central_asia": ["KZ", "KG", "UZ", "TJ", "TM"],
}

EXTRA_TERMS_BY_CULTURE = {
    "china": [
        "cantonese songs",
        "hakka songs",
        "hokkien songs",
        "taiwanese hokkien",
        "minnan songs",
        "teochew songs",
        "shanghainese songs",
        "sichuan dialect songs",
        "wu chinese songs",
        "yue chinese songs",
    ],
}


def parse_targets(spec: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid target item {part!r}; expected culture=count")
        culture, value = part.split("=", 1)
        culture = culture.strip()
        targets[culture] = int(value.strip())
    return targets


def read_unique_rows(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.exists():
        return []
    by_id: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            track_id = str(row.get("track_id", "")).strip()
            if track_id and track_id not in by_id:
                by_id[track_id] = row
    return list(by_id.values())


def read_unique_rows_many(metadata_paths: list[Path]) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                track_id = str(row.get("track_id", "")).strip()
                if track_id and track_id not in by_id:
                    by_id[track_id] = row
    return list(by_id.values())


def read_state_collected(state_path: Path, fallback: int) -> int:
    if not state_path.exists():
        return fallback
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    return max(int(state.get("total_collected", 0)), fallback)


def countries_by_culture() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for country, culture in COUNTRY_TO_CULTURE.items():
        grouped.setdefault(culture, []).append(country)
    grouped.update({culture: list(countries) for culture, countries in CUSTOM_CULTURE_COUNTRIES.items()})
    return grouped


def culture_counts(out_dir: Path, count_metadata: list[Path]) -> tuple[Counter[str], int]:
    rows = read_unique_rows_many([out_dir / "metadata.csv", *count_metadata])
    counts = Counter(str(row.get("culture", "")).strip() for row in rows)
    return counts, len(rows)


def deficits(counts: Counter[str], targets: dict[str, int]) -> dict[str, int]:
    return {culture: max(0, target - counts.get(culture, 0)) for culture, target in targets.items()}


def choose_next_culture(counts: Counter[str], targets: dict[str, int]) -> str | None:
    for culture, target in targets.items():
        if counts.get(culture, 0) < target:
            return culture
    return None


def write_status(
    status_path: Path,
    *,
    counts: Counter[str],
    targets: dict[str, int],
    selected_culture: str | None,
    last_exit_code: int | None,
    idle_rounds: int,
    attempts: int,
) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "targets": targets,
        "counts": {culture: counts.get(culture, 0) for culture in targets},
        "deficits": deficits(counts, targets),
        "selected_culture": selected_culture,
        "last_exit_code": last_exit_code,
        "idle_rounds": idle_rounds,
        "attempts": attempts,
    }
    tmp = status_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(status_path)


def stream_subprocess(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[COMMAND] {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def merge_metadata(python_executable: str, out_dir: Path, merge_out: Path, log_dir: Path) -> int:
    inputs = [out_dir / "metadata.csv"]
    jamendo = Path("storage/public/jamendo_crawl/metadata.csv")
    if jamendo.exists():
        inputs.append(jamendo)
    cmd = [
        python_executable,
        "-m",
        "dcas.scripts.merge_metadata_dedup",
        "--inputs",
        *[str(p) for p in inputs],
        "--out",
        str(merge_out),
        "--require_audio_exists",
    ]
    log_path = log_dir / f"merge_after_itunes_supervisor_{time.strftime('%Y%m%d_%H%M%S')}.log"
    return stream_subprocess(cmd, log_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervise iTunes culture top-up crawls.")
    ap.add_argument("--out_dir", default="./storage/public/itunes_crawl")
    ap.add_argument("--targets", default=",".join(f"{k}={v}" for k, v in DEFAULT_TARGETS.items()))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint_interval", type=int, default=300)
    ap.add_argument("--max_per_query", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=120)
    ap.add_argument("--restart_delay", type=int, default=120)
    ap.add_argument("--max_attempts", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--idle_round_limit", type=int, default=8, help="0 means unlimited")
    ap.add_argument(
        "--count_metadata",
        nargs="*",
        default=[],
        help="Extra metadata.csv files to count toward culture targets",
    )
    ap.add_argument("--merge_out", default="./storage/public/merged/metadata_merged.csv")
    ap.add_argument("--skip_merge", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--python_executable", default=sys.executable)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "itunes_nonwestern_supervisor_status.json"
    pid_path = out_dir / "itunes_nonwestern_supervisor.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    targets = parse_targets(args.targets)
    count_metadata = [Path(p) for p in args.count_metadata if Path(p).exists()]
    grouped = countries_by_culture()
    unknown = [culture for culture in targets if culture not in grouped]
    if unknown:
        print(f"[ERROR] No iTunes countries mapped for cultures: {unknown}")
        return 2

    attempts = 0
    idle_rounds = 0
    last_exit_code: int | None = None

    print("[SUPERVISOR START]")
    print(f"out_dir={out_dir}")
    print(f"targets={targets}")
    print(f"count_metadata={[str(p) for p in count_metadata]}")

    while True:
        counts, unique_total = culture_counts(out_dir, count_metadata)
        selected = choose_next_culture(counts, targets)
        write_status(
            status_path,
            counts=counts,
            targets=targets,
            selected_culture=selected,
            last_exit_code=last_exit_code,
            idle_rounds=idle_rounds,
            attempts=attempts,
        )
        print(f"[STATUS] total={unique_total} counts={dict((k, counts.get(k, 0)) for k in targets)}")

        if selected is None:
            print("[TARGETS MET] All requested iTunes culture targets are satisfied.")
            if not args.skip_merge:
                merge_out = Path(args.merge_out)
                merge_out.parent.mkdir(parents=True, exist_ok=True)
                last_exit_code = merge_metadata(args.python_executable, out_dir, merge_out, log_dir)
                print(f"[MERGE END] exit={last_exit_code}")
                return last_exit_code
            return 0

        if args.max_attempts and attempts >= args.max_attempts:
            print(f"[STOP] max_attempts reached: {args.max_attempts}")
            return 3

        attempts += 1
        before_selected = counts.get(selected, 0)
        before_total = unique_total
        remaining = targets[selected] - before_selected
        batch = max(1, min(args.batch_size, remaining))
        itunes_total = len(read_unique_rows(out_dir / "metadata.csv"))
        state_total = read_state_collected(out_dir / "state.json", itunes_total)
        target_total = state_total + batch
        countries = grouped[selected]
        country_arg = ",".join(countries)
        extra_terms = EXTRA_TERMS_BY_CULTURE.get(selected, [])
        log_path = log_dir / (f"itunes_nonwestern_{selected}_{time.strftime('%Y%m%d_%H%M%S')}_attempt{attempts}.log")

        cmd = [
            args.python_executable,
            "-m",
            "dcas.scripts.crawl_itunes_previews",
            "--out_dir",
            str(out_dir),
            "--countries",
            country_arg,
            "--target_total",
            str(target_total),
            "--workers",
            str(args.workers),
            "--checkpoint_interval",
            str(args.checkpoint_interval),
            "--max_per_query",
            str(args.max_per_query),
            "--culture_override",
            selected,
            "--resume",
        ]
        if extra_terms:
            cmd.extend(["--extra_terms", ",".join(extra_terms)])

        print(
            f"[ATTEMPT {attempts}] culture={selected} "
            f"current={before_selected} target={targets[selected]} "
            f"batch={batch} countries={country_arg}"
        )
        if args.dry_run:
            print("[DRY RUN]", " ".join(cmd))
            return 0

        last_exit_code = stream_subprocess(cmd, log_path)
        counts_after, total_after = culture_counts(out_dir, count_metadata)
        after_selected = counts_after.get(selected, 0)
        gained = after_selected - before_selected
        total_gained = total_after - before_total
        print(f"[ATTEMPT END] exit={last_exit_code} culture_gain={gained} total_gain={total_gained}")

        if gained <= 0 and total_gained <= 0:
            idle_rounds += 1
            print(f"[IDLE] No metadata growth in this attempt. idle_rounds={idle_rounds}")
        else:
            idle_rounds = 0

        if args.idle_round_limit and idle_rounds >= args.idle_round_limit:
            print(f"[STOP] idle_round_limit reached: {args.idle_round_limit}")
            return 4

        if last_exit_code != 0:
            print(f"[RESTART WAIT] crawler exited {last_exit_code}; sleeping {args.restart_delay}s")
            time.sleep(args.restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
