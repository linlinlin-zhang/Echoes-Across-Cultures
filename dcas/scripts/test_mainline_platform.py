from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dcas_server.mainline_platform import MainlineWeights, get_mainline_platform
from dcas_server.paths import Storage


def _write_markdown(report: dict[str, Any], out_path: Path) -> None:
    seed = report["seeds"][0] if report.get("seeds") else {}
    lines: list[str] = []
    lines.append("# Mainline Platform Recommendation Smoke Test")
    lines.append("")
    lines.append("## Algorithm")
    algo = report.get("algorithm", {})
    lines.append(f"- name: `{algo.get('name')}`")
    lines.append(f"- mode: `{algo.get('mode')}`")
    lines.append(f"- model: `{algo.get('model')}`")
    lines.append(f"- reranker: `{algo.get('reranker')}`")
    lines.append("")
    lines.append("## Seed")
    lines.append(f"- track_id: `{seed.get('track_id')}`")
    lines.append(f"- title: {seed.get('title')}")
    lines.append(f"- artist: {seed.get('artist')}")
    lines.append(f"- culture: `{seed.get('culture')}`")
    lines.append(f"- source: `{seed.get('source_dataset')}`")
    lines.append(f"- platform_url: {seed.get('platform_track_url')}")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("| rank | track_id | title | artist | culture | source | score | platform_url |")
    lines.append("|---:|---|---|---|---|---|---:|---|")
    for item in report.get("recommendations", []):
        lines.append(
            "| {rank} | `{track_id}` | {title} | {artist} | `{culture}` | `{source}` | {score:.6f} | {url} |".format(
                rank=int(item.get("rank") or 0),
                track_id=item.get("track_id") or "",
                title=str(item.get("title") or "").replace("|", "\\|"),
                artist=str(item.get("artist") or "").replace("|", "\\|"),
                culture=item.get("culture") or "",
                source=item.get("source_dataset") or "",
                score=float(item.get("score") or 0.0),
                url=item.get("platform_track_url") or "",
            )
        )
    lines.append("")
    lines.append("## Metrics")
    lines.append("```json")
    lines.append(json.dumps(report.get("metrics", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Warnings")
    for warning in report.get("warnings", []):
        lines.append(f"- {warning}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(
    *,
    out_json: Path,
    out_md: Path,
    mode: str,
    k: int,
    seed_track_id: str | None,
    seed_culture: str | None,
    target_culture: str | None,
    random_seed: int,
    prefer_cuda: bool,
) -> dict[str, Any]:
    platform = get_mainline_platform(Storage(root=Path("storage")), prefer_cuda=prefer_cuda)
    report = platform.recommend(
        seed_track_ids=[seed_track_id] if seed_track_id else [],
        seed_culture=seed_culture,
        target_culture=target_culture,
        mode=mode,
        k=k,
        recall_k=max(600, 60 * int(k)),
        random_seed=random_seed,
        weights=MainlineWeights(),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, out_md)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the cached mainline recommendation platform.")
    ap.add_argument("--out_json", default="reports/mainline_platform_smoke_20260524.json")
    ap.add_argument("--out_md", default="reports/mainline_platform_smoke_20260524.md")
    ap.add_argument("--mode", default="open", choices=["open", "target"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed_track_id", default=None)
    ap.add_argument("--seed_culture", default="caribbean")
    ap.add_argument("--target_culture", default=None)
    ap.add_argument("--random_seed", type=int, default=20260524)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    report = run_smoke(
        out_json=Path(args.out_json),
        out_md=Path(args.out_md),
        mode=str(args.mode),
        k=int(args.k),
        seed_track_id=args.seed_track_id,
        seed_culture=args.seed_culture,
        target_culture=args.target_culture,
        random_seed=int(args.random_seed),
        prefer_cuda=bool(args.prefer_cuda),
    )
    seed = report["seeds"][0]
    print(f"[OK] seed={seed['track_id']} {seed['title']} ({seed['culture']})")
    for item in report["recommendations"][: int(args.k)]:
        print(
            f"{item['rank']:02d}. {item['title']} - {item['artist']} "
            f"[{item['culture']}/{item['source_dataset']}] score={item['score']:.4f}"
        )
    print(f"[REPORT] {Path(args.out_json).resolve()}")
    print(f"[REPORT] {Path(args.out_md).resolve()}")


if __name__ == "__main__":
    main()
