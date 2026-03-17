from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_similar(value: str) -> bool | None:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "similar", "sim"}:
        return True
    if v in {"0", "false", "no", "n", "dissimilar", "diff"}:
        return False
    return None


def build_constraints_from_annotations(annotations_csv: str, out_path: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    skipped = 0
    with open(annotations_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = str(row.get("track_id_a", "")).strip()
            b = str(row.get("track_id_b", "")).strip()
            similar = _parse_similar(str(row.get("similar", "")))
            rationale = str(row.get("rationale", "")).strip()
            if not a or not b or a == b or similar is None:
                skipped += 1
                continue
            key_a, key_b = (a, b) if a < b else (b, a)
            rows.append(
                {
                    "track_id_a": key_a,
                    "track_id_b": key_b,
                    "similar": bool(similar),
                    "rationale": rationale,
                }
            )

    dedup: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        dedup[(str(row["track_id_a"]), str(row["track_id_b"]))] = row
    merged = list(dedup.values())
    merged.sort(key=lambda r: (str(r["track_id_a"]), str(r["track_id_b"])))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_positive = sum(1 for row in merged if bool(row["similar"]))
    return {
        "annotations_csv": str(annotations_csv),
        "out_path": str(out),
        "n_constraints": int(len(merged)),
        "n_positive": int(n_positive),
        "n_negative": int(len(merged) - n_positive),
        "n_skipped": int(skipped),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pairwise constraints jsonl from annotated PAL CSV.")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rep = build_constraints_from_annotations(annotations_csv=str(args.annotations), out_path=str(args.out))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
