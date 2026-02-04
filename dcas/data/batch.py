from __future__ import annotations

from typing import Sequence

import torch

from dcas.utils import Batch


def collate_batch(items: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]) -> Batch:
    xs, cultures, track_indices, affects = zip(*items)
    x = torch.stack(xs, dim=0)
    culture = torch.stack(cultures, dim=0)
    track_index = torch.stack(track_indices, dim=0)
    affect_label: torch.Tensor | None
    if affects[0] is None:
        affect_label = None
    else:
        affect_label = torch.stack([a for a in affects if a is not None], dim=0)
    return Batch(x=x, culture=culture, track_index=track_index, affect_label=affect_label)

