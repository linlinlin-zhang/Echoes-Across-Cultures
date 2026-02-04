from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PairwiseConstraint:
    track_id_a: str
    track_id_b: str
    similar: bool
    rationale: str | None = None


def load_constraints(path: str) -> list[PairwiseConstraint]:
    out: list[PairwiseConstraint] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                PairwiseConstraint(
                    track_id_a=str(obj["track_id_a"]),
                    track_id_b=str(obj["track_id_b"]),
                    similar=bool(obj["similar"]),
                    rationale=str(obj["rationale"]) if obj.get("rationale") is not None else None,
                )
            )
    return out


def save_constraints(path: str, constraints: list[PairwiseConstraint]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in constraints:
            f.write(
                json.dumps(
                    {
                        "track_id_a": c.track_id_a,
                        "track_id_b": c.track_id_b,
                        "similar": c.similar,
                        "rationale": c.rationale,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

