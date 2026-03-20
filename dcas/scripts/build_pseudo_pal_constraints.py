from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dcas.pal.constraints import PairwiseConstraint, save_constraints


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "" if text in {"", "nan", "none", "null"} else text


def _load_rows(metadata_csv: str | Path) -> list[dict[str, str]]:
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError("metadata is empty")
    return rows


def _row_view(row: dict[str, str]) -> dict[str, str]:
    return {
        "track_id": str(row.get("track_id", "")).strip(),
        "culture": _norm(row.get("culture")),
        "source_dataset": _norm(row.get("source_dataset")),
        "coarse_label": _norm(row.get("coarse_label")),
        "era": _norm(row.get("era")),
        "instrument_family": _norm(row.get("instrument_family")),
        "language": _norm(row.get("language")),
        "substyle": _norm(row.get("substyle")),
        "is_instrumental": _norm(row.get("is_instrumental")),
    }


def _positive_candidate(a: dict[str, str], b: dict[str, str]) -> tuple[float, str] | None:
    if a["track_id"] == b["track_id"]:
        return None
    score = 0.0
    reasons: list[str] = []

    if a["coarse_label"] != "" and a["coarse_label"] == b["coarse_label"]:
        score += 3.0
        reasons.append(f"same coarse_label={a['coarse_label']}")
    else:
        return None

    if a["era"] != "" and a["era"] == b["era"]:
        score += 2.0
        reasons.append(f"same era={a['era']}")
    if a["is_instrumental"] != "" and a["is_instrumental"] == b["is_instrumental"]:
        score += 1.5
        reasons.append(f"same is_instrumental={a['is_instrumental']}")
    if a["instrument_family"] != "" and a["instrument_family"] == b["instrument_family"]:
        score += 1.0
        reasons.append(f"same instrument_family={a['instrument_family']}")
    if a["language"] != "" and a["language"] == b["language"]:
        score += 1.0
        reasons.append(f"same language={a['language']}")
    if a["substyle"] != "" and a["substyle"] == b["substyle"]:
        score += 1.0
        reasons.append(f"same substyle={a['substyle']}")
    if a["culture"] != "" and a["culture"] != b["culture"]:
        score += 1.0
        reasons.append(f"cross_culture={a['culture']}->{b['culture']}")
    if a["source_dataset"] != "" and a["source_dataset"] != b["source_dataset"]:
        score += 1.5
        reasons.append("cross_source")
    if a["language"] != "" and b["language"] != "" and a["language"] != b["language"]:
        score -= 0.5
    if a["is_instrumental"] != "" and b["is_instrumental"] != "" and a["is_instrumental"] != b["is_instrumental"]:
        return None

    if score < 5.0:
        return None
    return score, "pseudo-positive: " + "; ".join(reasons)


def _negative_candidate(a: dict[str, str], b: dict[str, str]) -> tuple[float, str] | None:
    if a["track_id"] == b["track_id"]:
        return None

    score = 0.0
    reasons: list[str] = []
    if a["coarse_label"] != "" and b["coarse_label"] != "" and a["coarse_label"] != b["coarse_label"]:
        score += 3.0
        reasons.append(f"coarse_label {a['coarse_label']} vs {b['coarse_label']}")
    if a["era"] != "" and b["era"] != "" and a["era"] != b["era"]:
        score += 2.0
        reasons.append(f"era {a['era']} vs {b['era']}")
    if a["is_instrumental"] != "" and b["is_instrumental"] != "" and a["is_instrumental"] != b["is_instrumental"]:
        score += 2.0
        reasons.append(f"is_instrumental {a['is_instrumental']} vs {b['is_instrumental']}")
    if a["instrument_family"] != "" and b["instrument_family"] != "" and a["instrument_family"] != b["instrument_family"]:
        score += 1.0
        reasons.append(f"instrument_family {a['instrument_family']} vs {b['instrument_family']}")
    if a["language"] != "" and b["language"] != "" and a["language"] != b["language"]:
        score += 1.0
        reasons.append(f"language {a['language']} vs {b['language']}")
    if a["culture"] != "" and b["culture"] != "" and a["culture"] != b["culture"]:
        score += 0.5
        reasons.append(f"cross_culture={a['culture']}->{b['culture']}")
    if a["source_dataset"] != "" and b["source_dataset"] != "" and a["source_dataset"] != b["source_dataset"]:
        score += 0.25

    if score < 5.0:
        return None
    return score, "pseudo-negative: " + "; ".join(reasons)


def _positive_candidate_with_threshold(
    a: dict[str, str],
    b: dict[str, str],
    min_score: float,
) -> tuple[float, str] | None:
    if a["track_id"] == b["track_id"]:
        return None

    score = 0.0
    reasons: list[str] = []

    if a["coarse_label"] != "" and a["coarse_label"] == b["coarse_label"]:
        score += 3.0
        reasons.append(f"same coarse_label={a['coarse_label']}")
    else:
        return None

    if a["era"] != "" and a["era"] == b["era"]:
        score += 2.0
        reasons.append(f"same era={a['era']}")
    if a["is_instrumental"] != "" and a["is_instrumental"] == b["is_instrumental"]:
        score += 1.5
        reasons.append(f"same is_instrumental={a['is_instrumental']}")
    if a["instrument_family"] != "" and a["instrument_family"] == b["instrument_family"]:
        score += 1.0
        reasons.append(f"same instrument_family={a['instrument_family']}")
    if a["language"] != "" and a["language"] == b["language"]:
        score += 1.0
        reasons.append(f"same language={a['language']}")
    if a["substyle"] != "" and a["substyle"] == b["substyle"]:
        score += 1.0
        reasons.append(f"same substyle={a['substyle']}")
    if a["culture"] != "" and a["culture"] != b["culture"]:
        score += 1.0
        reasons.append(f"cross_culture={a['culture']}->{b['culture']}")
    if a["source_dataset"] != "" and a["source_dataset"] != b["source_dataset"]:
        score += 1.5
        reasons.append("cross_source")
    if a["language"] != "" and b["language"] != "" and a["language"] != b["language"]:
        score -= 0.5
    if a["is_instrumental"] != "" and b["is_instrumental"] != "" and a["is_instrumental"] != b["is_instrumental"]:
        return None

    if score < float(min_score):
        return None
    return score, "pseudo-positive: " + "; ".join(reasons)


def _negative_candidate_with_threshold(
    a: dict[str, str],
    b: dict[str, str],
    min_score: float,
) -> tuple[float, str] | None:
    if a["track_id"] == b["track_id"]:
        return None

    score = 0.0
    reasons: list[str] = []
    if a["coarse_label"] != "" and b["coarse_label"] != "" and a["coarse_label"] != b["coarse_label"]:
        score += 3.0
        reasons.append(f"coarse_label {a['coarse_label']} vs {b['coarse_label']}")
    if a["era"] != "" and b["era"] != "" and a["era"] != b["era"]:
        score += 2.0
        reasons.append(f"era {a['era']} vs {b['era']}")
    if a["is_instrumental"] != "" and b["is_instrumental"] != "" and a["is_instrumental"] != b["is_instrumental"]:
        score += 2.0
        reasons.append(f"is_instrumental {a['is_instrumental']} vs {b['is_instrumental']}")
    if a["instrument_family"] != "" and b["instrument_family"] != "" and a["instrument_family"] != b["instrument_family"]:
        score += 1.0
        reasons.append(f"instrument_family {a['instrument_family']} vs {b['instrument_family']}")
    if a["language"] != "" and b["language"] != "" and a["language"] != b["language"]:
        score += 1.0
        reasons.append(f"language {a['language']} vs {b['language']}")
    if a["culture"] != "" and b["culture"] != "" and a["culture"] != b["culture"]:
        score += 0.5
        reasons.append(f"cross_culture={a['culture']}->{b['culture']}")
    if a["source_dataset"] != "" and b["source_dataset"] != "" and a["source_dataset"] != b["source_dataset"]:
        score += 0.25

    if score < float(min_score):
        return None
    return score, "pseudo-negative: " + "; ".join(reasons)


def _select_pairs(
    candidates: list[tuple[float, PairwiseConstraint]],
    limit: int,
    per_track_cap: int,
) -> list[PairwiseConstraint]:
    selected: list[PairwiseConstraint] = []
    usage: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    for _, constraint in sorted(candidates, key=lambda x: (-float(x[0]), x[1].track_id_a, x[1].track_id_b)):
        key = (constraint.track_id_a, constraint.track_id_b)
        if key in seen:
            continue
        if usage[constraint.track_id_a] >= int(per_track_cap) or usage[constraint.track_id_b] >= int(per_track_cap):
            continue
        selected.append(constraint)
        seen.add(key)
        usage[constraint.track_id_a] += 1
        usage[constraint.track_id_b] += 1
        if len(selected) >= int(limit):
            break
    return selected


def build_pseudo_constraints(
    metadata_csv: str | Path,
    out_path: str | Path,
    n_positive: int = 800,
    n_negative: int = 800,
    per_track_cap: int = 6,
    positive_min_score: float = 5.0,
    negative_min_score: float = 5.0,
) -> dict[str, Any]:
    rows = [_row_view(r) for r in _load_rows(metadata_csv)]
    rows = [r for r in rows if r["track_id"] and r["culture"]]
    if not rows:
        raise RuntimeError("no usable rows in metadata")

    positive_candidates: list[tuple[float, PairwiseConstraint]] = []
    negative_candidates: list[tuple[float, PairwiseConstraint]] = []
    culture_pair_counts: defaultdict[str, int] = defaultdict(int)

    for i in range(len(rows)):
        a = rows[i]
        for j in range(i + 1, len(rows)):
            b = rows[j]
            culture_key = "|".join(sorted([a["culture"], b["culture"]]))
            culture_pair_counts[culture_key] += 1

            pos = _positive_candidate_with_threshold(a, b, min_score=float(positive_min_score))
            if pos is not None:
                score, rationale = pos
                positive_candidates.append(
                    (
                        float(score),
                        PairwiseConstraint(
                            track_id_a=min(a["track_id"], b["track_id"]),
                            track_id_b=max(a["track_id"], b["track_id"]),
                            similar=True,
                            rationale=rationale,
                        ),
                    )
                )

            neg = _negative_candidate_with_threshold(a, b, min_score=float(negative_min_score))
            if neg is not None:
                score, rationale = neg
                negative_candidates.append(
                    (
                        float(score),
                        PairwiseConstraint(
                            track_id_a=min(a["track_id"], b["track_id"]),
                            track_id_b=max(a["track_id"], b["track_id"]),
                            similar=False,
                            rationale=rationale,
                        ),
                    )
                )

    positive = _select_pairs(
        candidates=positive_candidates,
        limit=int(n_positive),
        per_track_cap=int(per_track_cap),
    )
    negative = _select_pairs(
        candidates=negative_candidates,
        limit=int(n_negative),
        per_track_cap=int(per_track_cap),
    )
    constraints = sorted(
        positive + negative,
        key=lambda c: (c.track_id_a, c.track_id_b, 0 if c.similar else 1),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_constraints(str(out_path), constraints)

    report = {
        "metadata_csv": str(Path(metadata_csv).resolve()),
        "out_path": str(Path(out_path).resolve()),
        "n_rows": int(len(rows)),
        "n_positive_candidates": int(len(positive_candidates)),
        "n_negative_candidates": int(len(negative_candidates)),
        "n_positive_selected": int(len(positive)),
        "n_negative_selected": int(len(negative)),
        "n_constraints": int(len(constraints)),
        "per_track_cap": int(per_track_cap),
        "positive_min_score": float(positive_min_score),
        "negative_min_score": float(negative_min_score),
        "culture_pair_candidates_top20": [
            {"culture_pair": key, "count": int(count)}
            for key, count in sorted(culture_pair_counts.items(), key=lambda x: (-x[1], x[0]))[:20]
        ],
        "positive_cross_culture_ratio": float(
            sum(1 for c in positive if "cross_culture=" in str(c.rationale or "")) / max(1, len(positive))
        ),
        "positive_cross_source_ratio": float(
            sum(1 for c in positive if "cross_source" in str(c.rationale or "")) / max(1, len(positive))
        ),
    }
    report_path = Path(out_path).with_suffix(Path(out_path).suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build high-confidence pseudo PAL constraints from unified metadata.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_positive", type=int, default=800)
    ap.add_argument("--n_negative", type=int, default=800)
    ap.add_argument("--per_track_cap", type=int, default=6)
    ap.add_argument("--positive_min_score", type=float, default=5.0)
    ap.add_argument("--negative_min_score", type=float, default=5.0)
    args = ap.parse_args()

    rep = build_pseudo_constraints(
        metadata_csv=str(args.metadata),
        out_path=str(args.out),
        n_positive=int(args.n_positive),
        n_negative=int(args.n_negative),
        per_track_cap=int(args.per_track_cap),
        positive_min_score=float(args.positive_min_score),
        negative_min_score=float(args.negative_min_score),
    )
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
