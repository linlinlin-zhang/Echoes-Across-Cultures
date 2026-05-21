from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_SITE = ROOT / "volunteer_site"
OUT_DIR = ROOT / "participant_versions"
KEY_PATH = ROOT / "researcher_private" / "method_key_private.csv"


def read_tasks() -> list[dict]:
    return json.loads((BASE_SITE / "tasks.json").read_text(encoding="utf-8"))


def read_key() -> dict[str, dict[str, str]]:
    with KEY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
      return {row["task_id"]: row for row in csv.DictReader(f)}


def write_tasks_js(site: Path, tasks: list[dict], participant_id: str) -> None:
    payload = (
        f'window.PILOT_DEFAULT_PARTICIPANT_ID = "{participant_id}";\n'
        + "window.PILOT_TASKS = "
        + json.dumps(tasks, ensure_ascii=True, indent=2)
        + ";\n"
    )
    (site / "tasks.js").write_text(payload, encoding="utf-8")
    (site / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def swap_task_ab(task: dict) -> dict:
    out = json.loads(json.dumps(task, ensure_ascii=False))
    out["candidate_a"], out["candidate_b"] = out["candidate_b"], out["candidate_a"]
    return out


def make_participant_tasks(base_tasks: list[dict], participant_idx: int) -> tuple[list[dict], dict[str, bool]]:
    rng = random.Random(20260418 + participant_idx)
    order = list(range(len(base_tasks)))
    rng.shuffle(order)
    tasks = []
    swaps: dict[str, bool] = {}
    for position, original_idx in enumerate(order):
        task = json.loads(json.dumps(base_tasks[original_idx], ensure_ascii=False))
        should_swap = (participant_idx + original_idx) % 2 == 1
        if should_swap:
            task = swap_task_ab(task)
        tasks.append(task)
        swaps[str(task["task_id"])] = should_swap
    return tasks, swaps


def write_participant_key(participant_id: str, base_key: dict[str, dict[str, str]], swaps: dict[str, bool]) -> None:
    rows = []
    for task_id, row in base_key.items():
        out = dict(row)
        out["participant_id"] = participant_id
        if swaps.get(task_id):
            out["candidate_a_track_id"], out["candidate_b_track_id"] = out["candidate_b_track_id"], out["candidate_a_track_id"]
            out["candidate_a_method"], out["candidate_b_method"] = out["candidate_b_method"], out["candidate_a_method"]
        rows.append(out)
    fieldnames = ["participant_id"] + [k for k in rows[0].keys() if k != "participant_id"]
    out_path = ROOT / "researcher_private" / f"method_key_private_{participant_id}.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_tasks = read_tasks()
    base_key = read_key()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, str]] = []
    for i in range(1, 11):
        participant_id = f"P{i:02d}"
        site = OUT_DIR / participant_id
        shutil.copytree(BASE_SITE, site)
        tasks, swaps = make_participant_tasks(base_tasks, i)
        write_tasks_js(site, tasks, participant_id)
        write_participant_key(participant_id, base_key, swaps)
        for task_id, row in base_key.items():
            out = dict(row)
            out["participant_id"] = participant_id
            if swaps.get(task_id):
                out["candidate_a_track_id"], out["candidate_b_track_id"] = out["candidate_b_track_id"], out["candidate_a_track_id"]
                out["candidate_a_method"], out["candidate_b_method"] = out["candidate_b_method"], out["candidate_a_method"]
            combined_rows.append(out)

        zip_base = OUT_DIR / f"{participant_id}_volunteer_site"
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=site)
        print(f"{participant_id}: {zip_path}")

    combined_path = ROOT / "researcher_private" / "method_key_private_counterbalanced_all.csv"
    fieldnames = ["participant_id"] + [k for k in combined_rows[0].keys() if k != "participant_id"]
    with combined_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"combined key: {combined_path}")


if __name__ == "__main__":
    main()
