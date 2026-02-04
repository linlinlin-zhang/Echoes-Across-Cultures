from __future__ import annotations

from pydantic import BaseModel, Field


class ToyGenerateRequest(BaseModel):
    name: str = Field(default="toy")
    n_tracks: int = Field(default=3000, ge=100)
    dim: int = Field(default=128, ge=8, le=4096)
    seed: int = Field(default=7)


class TrainRequest(BaseModel):
    tracks_path: str
    out_name: str = Field(default="model.pt")
    constraints_path: str | None = None
    epochs: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=256, ge=16, le=4096)
    lr: float = Field(default=2e-3, gt=0)
    seed: int = Field(default=42)
    prefer_cuda: bool = Field(default=False)
    lambda_constraints: float = Field(default=0.1, ge=0)
    constraint_margin: float = Field(default=1.0, gt=0)


class RecommendRequest(BaseModel):
    model_path: str
    tracks_path: str
    interactions_path: str
    user_id: str
    target_culture: str
    k: int = Field(default=20, ge=1, le=200)
    prefer_cuda: bool = Field(default=False)
    epsilon: float = Field(default=0.1, gt=0)
    iters: int = Field(default=200, ge=10, le=2000)


class PalRequest(BaseModel):
    model_path: str
    tracks_path: str
    out_name: str = Field(default="pal_tasks.jsonl")
    n: int = Field(default=100, ge=1, le=2000)
    prefer_cuda: bool = Field(default=False)

