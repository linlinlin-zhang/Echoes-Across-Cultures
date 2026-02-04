from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Tracks:
    track_id: np.ndarray
    culture: np.ndarray
    embedding: np.ndarray
    affect_label: np.ndarray | None

    def __len__(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embedding.shape[1])

    def cultures(self) -> list[str]:
        return sorted({str(x) for x in self.culture.tolist()})

    def indices_of_cultures(self, cultures: Iterable[str]) -> np.ndarray:
        target = {str(c) for c in cultures}
        mask = np.array([str(x) in target for x in self.culture.tolist()], dtype=bool)
        return np.nonzero(mask)[0]


def load_tracks(path: str) -> Tracks:
    data = np.load(path, allow_pickle=False)
    track_id = data["track_id"].astype(str)
    culture = data["culture"].astype(str)
    embedding = data["embedding"].astype(np.float32)
    affect_label = data["affect_label"].astype(np.int64) if "affect_label" in data else None
    if embedding.ndim != 2:
        raise ValueError("embedding must be 2D (N, D)")
    if track_id.shape[0] != embedding.shape[0] or culture.shape[0] != embedding.shape[0]:
        raise ValueError("track_id/culture/embedding must have same N")
    if affect_label is not None and affect_label.shape[0] != embedding.shape[0]:
        raise ValueError("affect_label must have same N as embedding")
    return Tracks(track_id=track_id, culture=culture, embedding=embedding, affect_label=affect_label)

