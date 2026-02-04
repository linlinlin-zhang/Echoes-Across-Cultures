from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .npz_tracks import Tracks


@dataclass(frozen=True)
class CultureVocab:
    culture_to_id: dict[str, int]
    id_to_culture: list[str]

    @classmethod
    def from_tracks(cls, tracks: Tracks) -> "CultureVocab":
        id_to_culture = tracks.cultures()
        culture_to_id = {c: i for i, c in enumerate(id_to_culture)}
        return cls(culture_to_id=culture_to_id, id_to_culture=id_to_culture)

    @classmethod
    def from_dict(cls, obj: dict) -> "CultureVocab":
        id_to_culture = [str(x) for x in obj["id_to_culture"]]
        culture_to_id = {str(k): int(v) for k, v in obj["culture_to_id"].items()}
        return cls(culture_to_id=culture_to_id, id_to_culture=id_to_culture)

    def to_dict(self) -> dict:
        return {
            "culture_to_id": dict(self.culture_to_id),
            "id_to_culture": list(self.id_to_culture),
        }

    def encode(self, culture: np.ndarray) -> np.ndarray:
        return np.array([self.culture_to_id[str(x)] for x in culture.tolist()], dtype=np.int64)


class TrackDataset(Dataset):
    def __init__(self, tracks: Tracks, vocab: CultureVocab):
        self.tracks = tracks
        self.vocab = vocab
        self._culture_id = vocab.encode(tracks.culture)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.tracks.embedding[idx])
        culture_id = torch.tensor(self._culture_id[idx], dtype=torch.long)
        track_index = torch.tensor(idx, dtype=torch.long)
        affect = None
        if self.tracks.affect_label is not None:
            affect = torch.tensor(int(self.tracks.affect_label[idx]), dtype=torch.long)
        return x, culture_id, track_index, affect

