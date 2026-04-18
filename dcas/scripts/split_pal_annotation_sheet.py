from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_tasks(tasks_path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (str(obj.get("track_id", "")).strip(), str(obj.get("compare_to", "")).strip())
            if key[0] and key[1]:
                out[key] = obj
    return out


def _load_sheet(sheet_path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(sheet_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _assignment_readme(
    out_dir: Path,
    annotators: list[str],
    summary: dict[str, Any],
    quick_guide_name: str,
) -> str:
    lines = [
        "# PAL Annotator Packets",
        "",
        "This directory contains ready-to-send annotation packets.",
        "",
        "Files to distribute:",
    ]
    for annotator in annotators:
        info = summary["annotators"][annotator]
        lines.append(
            f"- `{annotator}` -> `{Path(info['csv_path']).name}` ({int(info['n_tasks'])} tasks)"
        )
    lines.extend(
        [
            "",
            "Shared files:",
            f"- quick guide: `{quick_guide_name}`",
            "- assignment master: `tasks_round1_200_assignment_master.csv`",
            "- assignment summary: `assignment_summary.json`",
            "",
            "Recommended use:",
            "1. Send the quick guide to every annotator.",
            "2. Send each annotator only their own CSV.",
            "3. Ask them to keep the prefilled `annotator` column unchanged.",
            "4. Ask them to return the completed file with the original filename preserved.",
            "",
        ]
    )
    return "\n".join(lines)


def _quick_guide(annotators: list[str], summary: dict[str, Any]) -> str:
    lines = [
        "# 真人 PAL 标注说明简版",
        "",
        "这是一份给标注员的简版说明，目标是让大家拿到表后就能直接开始。",
        "",
        "## 1. 你要判断的不是“国家是不是一样”",
        "",
        "请不要根据国家、语言、乐器名、标签名直接下结论。",
        "你要判断的是：这两首曲子是否适合放在同一个歌单，或者出现在相近的听歌场景里。",
        "",
        "更接近下面这些问题：",
        "- 它们是不是都适合安静放松、专注工作或沉浸式聆听？",
        "- 它们是不是都更像庆典、舞动、热闹聚会这类场景？",
        "- 它们带给人的整体感觉、情绪基调和节奏能量是不是接近？",
        "- 如果把它们放进同一个歌单里，听起来会不会显得自然？",
        "",
        "## 2. 每对样本怎么填",
        "",
        "- `similar`：填 `yes` 或 `no`",
        "- `rationale`：写一句很短的理由",
        "- `annotator`：已经预填，不用改",
        "- `notes`：如果这对很难判断，在这里写原因",
        "",
        "## 3. 推荐的标注习惯",
        "",
        "- 戴耳机",
        "- 在安静环境里听",
        "- 音量尽量固定",
        "- 每对至少听一遍，必要时回放一次",
        "- 不确定时宁可留空并写说明，也不要硬猜",
        "",
        "## 4. 返回文件时怎么做",
        "",
        "- 直接在收到的 CSV 里填写",
        "- 不要改列名",
        "- 尽量保留原文件名返回",
        "",
        "## 5. 这轮分发表",
        "",
        f"本轮共分给 {len(annotators)} 位标注员，总计 {int(summary['n_total_tasks'])} 对任务。",
    ]
    for annotator in annotators:
        info = summary["annotators"][annotator]
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(info["selection_group_counts"].items()))
        lines.append(f"- `{annotator}`：{int(info['n_tasks'])} 对；文化分布：{counts}")
    lines.append("")
    lines.append("如果先做试标，建议先用 `pilot_tasks_20_annotation.csv` 对齐理解，再进入各自的 50 对正式任务。")
    lines.append("")
    return "\n".join(lines)


def split_pal_annotation_sheet(
    tasks_path: str | Path,
    sheet_path: str | Path,
    out_dir: str | Path,
    annotators: list[str],
) -> dict[str, Any]:
    task_map = _load_tasks(tasks_path)
    sheet_rows, input_fieldnames = _load_sheet(sheet_path)
    if not annotators:
        raise RuntimeError("annotators list is empty")

    extra_fields = ["selection_group", "selection_rank_in_group", "assigned_annotator"]
    fieldnames = list(input_fieldnames)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    enriched_rows: list[dict[str, str]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_task_meta = 0

    for row in sheet_rows:
        key = (str(row.get("track_id_a", "")).strip(), str(row.get("track_id_b", "")).strip())
        task = task_map.get(key, {})
        if not task:
            missing_task_meta += 1
        enriched = dict(row)
        enriched["selection_group"] = str(task.get("selection_group", row.get("culture_a", ""))).strip()
        enriched["selection_rank_in_group"] = str(task.get("selection_rank_in_group", "")).strip()
        enriched["assigned_annotator"] = ""
        grouped[enriched["selection_group"]].append(enriched)
        enriched_rows.append(enriched)

    annotator_rows: dict[str, list[dict[str, str]]] = {name: [] for name in annotators}
    for group in sorted(grouped.keys()):
        group_rows = grouped[group]
        group_rows.sort(
            key=lambda r: (
                int(str(r.get("selection_rank_in_group", "") or "999999")),
                -float(str(r.get("uncertainty", "0") or "0")),
                str(r.get("task_id", "")),
            )
        )
        for idx, row in enumerate(group_rows):
            annotator = annotators[idx % len(annotators)]
            row["assigned_annotator"] = annotator
            row["annotator"] = annotator
            annotator_rows[annotator].append(row)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    master_path = out_dir_p / "tasks_round1_200_assignment_master.csv"
    _write_csv(master_path, enriched_rows, fieldnames)

    summary: dict[str, Any] = {
        "tasks_path": str(Path(tasks_path).resolve()),
        "sheet_path": str(Path(sheet_path).resolve()),
        "out_dir": str(out_dir_p.resolve()),
        "annotators": {},
        "n_total_tasks": int(len(enriched_rows)),
        "n_annotators": int(len(annotators)),
        "missing_task_meta": int(missing_task_meta),
        "master_csv": str(master_path.resolve()),
    }

    for annotator in annotators:
        rows = sorted(
            annotator_rows[annotator],
            key=lambda r: (
                str(r.get("selection_group", "")),
                int(str(r.get("selection_rank_in_group", "") or "999999")),
                str(r.get("task_id", "")),
            ),
        )
        csv_path = out_dir_p / f"{annotator}_tasks_round1_{len(rows)}_annotation.csv"
        _write_csv(csv_path, rows, fieldnames)
        group_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            group_counts[str(row.get("selection_group", ""))] += 1
        summary["annotators"][annotator] = {
            "csv_path": str(csv_path.resolve()),
            "n_tasks": int(len(rows)),
            "selection_group_counts": dict(sorted(group_counts.items())),
        }

    summary_path = out_dir_p / "assignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(summary_path.resolve())

    quick_guide_path = out_dir_p / "ANNOTATION_QUICK_GUIDE_CN.md"
    _write_text(quick_guide_path, _quick_guide(annotators=annotators, summary=summary))
    summary["quick_guide"] = str(quick_guide_path.resolve())

    readme_path = out_dir_p / "README.md"
    _write_text(
        readme_path,
        _assignment_readme(
            out_dir=out_dir_p,
            annotators=annotators,
            summary=summary,
            quick_guide_name=quick_guide_path.name,
        ),
    )
    summary["readme"] = str(readme_path.resolve())

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Split a PAL annotation sheet into balanced per-annotator packets.")
    ap.add_argument("--tasks", required=True, help="Selected PAL tasks jsonl, ideally tasks_round1_200.jsonl")
    ap.add_argument("--sheet", required=True, help="Annotation CSV derived from the tasks file")
    ap.add_argument("--out_dir", required=True, help="Directory for annotator packets")
    ap.add_argument("--annotators", nargs="+", required=True, help="Annotator ids, e.g. annotator_A annotator_B")
    args = ap.parse_args()

    out = split_pal_annotation_sheet(
        tasks_path=str(args.tasks),
        sheet_path=str(args.sheet),
        out_dir=str(args.out_dir),
        annotators=[str(x) for x in args.annotators],
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
