from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class Interaction:
    user_id: str
    track_id: str
    weight: float


def load_interactions(path: str) -> list[Interaction]:
    out: list[Interaction] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = str(row["user_id"])
            track_id = str(row["track_id"])
            weight = float(row.get("weight", 1.0))
            out.append(Interaction(user_id=user_id, track_id=track_id, weight=weight))
    return out

