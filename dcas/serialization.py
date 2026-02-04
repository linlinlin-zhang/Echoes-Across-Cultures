from __future__ import annotations

from dataclasses import asdict

import torch

from dcas.data.torch_dataset import CultureVocab
from dcas.models.dcas_vae import DCASConfig, DCASModel


def save_checkpoint(path: str, model: DCASModel, vocab: CultureVocab) -> None:
    obj = {
        "cfg": asdict(model.cfg),
        "state_dict": model.state_dict(),
        "culture_vocab": vocab.to_dict(),
    }
    torch.save(obj, path)


def load_checkpoint(path: str, map_location: str | None = None) -> tuple[DCASModel, CultureVocab]:
    obj = torch.load(path, map_location=map_location)
    cfg = DCASConfig(**obj["cfg"])
    model = DCASModel(cfg)
    model.load_state_dict(obj["state_dict"])
    vocab = CultureVocab.from_dict(obj["culture_vocab"])
    return model, vocab

