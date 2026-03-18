from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_tasks(tasks_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def _normalize_group(value: object) -> str:
    group = str(value or "").strip()
    return group if group else "__unknown__"


def _allocate(group_sizes: dict[str, int], n_total: int) -> dict[str, int]:
    groups = sorted(group_sizes.keys())
    if not groups or int(n_total) <= 0:
        return {}
    quota = int(n_total) // len(groups)
    alloc = {g: min(quota, int(group_sizes[g])) for g in groups}
    used = sum(alloc.values())
    if used >= int(n_total):
        return alloc

    # Fill remaining budget by cycling through groups with spare capacity.
    while used < int(n_total):
        progressed = False
        for g in groups:
            if used >= int(n_total):
                break
            if alloc[g] < int(group_sizes[g]):
                alloc[g] += 1
                used += 1
                progressed = True
        if not progressed:
            break
    return alloc


def select_pal_tasks_stratified(
    tasks_path: str | Path,
    out_path: str | Path,
    n_total: int = 200,
    group_field: str = "culture",
    pool_multiplier: int = 3,
    seed: int = 42,
) -> dict[str, object]:
    rows = _load_tasks(str(tasks_path))
    if not rows:
        raise RuntimeError("tasks file is empty")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_normalize_group(row.get(group_field, ""))].append(row)

    group_sizes = {g: len(v) for g, v in grouped.items()}
    alloc = _allocate(group_sizes=group_sizes, n_total=int(n_total))
    rng = np.random.default_rng(int(seed))

    selected: list[dict[str, object]] = []
    summary_groups: list[dict[str, object]] = []
    for group in sorted(grouped.keys()):
        quota = int(alloc.get(group, 0))
        group_rows = sorted(grouped[group], key=lambda r: float(r.get("uncertainty", 0.0)), reverse=True)
        if quota <= 0 or not group_rows:
            summary_groups.append(
                {
                    "group": group,
                    "available": int(len(group_rows)),
                    "selected": 0,
                }
            )
            continue
        pool_size = min(len(group_rows), max(quota, quota * int(pool_multiplier)))
        pool = group_rows[:pool_size]
        if quota >= len(pool):
            picked = pool
        else:
            idx = rng.choice(len(pool), size=quota, replace=False)
            picked = [pool[int(i)] for i in sorted(idx.tolist())]
        picked = sorted(picked, key=lambda r: float(r.get("uncertainty", 0.0)), reverse=True)
        for rank, row in enumerate(picked, start=1):
            out_row = dict(row)
            out_row["selection_group"] = group
            out_row["selection_rank_in_group"] = int(rank)
            selected.append(out_row)
        summary_groups.append(
            {
                "group": group,
                "available": int(len(group_rows)),
                "selected": int(len(picked)),
                "pool_size": int(pool_size),
            }
        )

    selected = sorted(
        selected,
        key=lambda r: (
            str(r.get("selection_group", "")),
            -float(r.get("uncertainty", 0.0)),
            str(r.get("track_id", "")),
        ),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "tasks_path": str(Path(tasks_path).resolve()),
        "out_path": str(out_path.resolve()),
        "group_field": str(group_field),
        "n_input": int(len(rows)),
        "n_selected": int(len(selected)),
        "pool_multiplier": int(pool_multiplier),
        "seed": int(seed),
        "groups": summary_groups,
    }
    report_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["summary_json"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select a balanced subset of PAL tasks by culture/domain from a larger uncertainty-ranked candidate pool."
    )
    ap.add_argument("--tasks", required=True, help="Candidate PAL tasks jsonl")
    ap.add_argument("--out", required=True, help="Selected PAL tasks jsonl")
    ap.add_argument("--n_total", type=int, default=200, help="Total number of selected tasks")
    ap.add_argument("--group_field", default="culture", help="Task field used for balancing, default: culture")
    ap.add_argument(
        "--pool_multiplier",
        type=int,
        default=3,
        help="Sample each group's quota from its top quota*pool_multiplier uncertain tasks",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = select_pal_tasks_stratified(
        tasks_path=str(args.tasks),
        out_path=str(args.out),
        n_total=int(args.n_total),
        group_field=str(args.group_field),
        pool_multiplier=int(args.pool_multiplier),
        seed=int(args.seed),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
