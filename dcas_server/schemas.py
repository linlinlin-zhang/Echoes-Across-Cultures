from __future__ import annotations

from pydantic import BaseModel, Field


class ToyGenerateRequest(BaseModel):
    name: str = Field(default="toy")
    n_tracks: int = Field(default=3000, ge=100)
    dim: int = Field(default=128, ge=8, le=4096)
    seed: int = Field(default=7)


class DatasetBuildRequest(BaseModel):
    metadata_path: str
    out_name: str = Field(default="tracks.npz")
    model_id: str = Field(default="ntua-slp/CultureMERT-95M")
    device: str | None = None
    pooling: str = Field(default="mean")
    max_seconds: float = Field(default=30.0, gt=0)
    limit: int | None = Field(default=None, ge=1)
    skip_errors: bool = Field(default=False)


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
    lambda_domain: float = Field(default=0.5, ge=0)
    lambda_contrast: float = Field(default=0.2, ge=0)
    lambda_cov: float = Field(default=0.05, ge=0)
    lambda_tc: float = Field(default=0.05, ge=0)
    lambda_hsic: float = Field(default=0.02, ge=0)
    beta_kl: float = Field(default=1.0, ge=0)
    shared_encoder: bool = Field(default=False)
    regularizer_warmup_epochs: int = Field(default=0, ge=0, le=500)


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


class MainlineRecommendRequest(BaseModel):
    seed_track_id: str | None = None
    seed_track_ids: list[str] = Field(default_factory=list)
    seed_culture: str | None = None
    target_culture: str | None = None
    mode: str = Field(default="open")
    k: int = Field(default=10, ge=1, le=100)
    recall_k: int = Field(default=600, ge=10, le=5000)
    random_seed: int | None = Field(default=42)
    prefer_cuda: bool = Field(default=False)
    exclude_same_artist: bool = Field(default=False)
    exclude_low_signal: bool = Field(default=True)
    relevance_weight: float = Field(default=0.48, ge=0.0)
    novelty_weight: float = Field(default=0.10, ge=0.0)
    target_affinity_weight: float = Field(default=0.22, ge=0.0)
    minority_weight: float = Field(default=0.14, ge=0.0)
    source_weight: float = Field(default=0.06, ge=0.0)
    diversity_lambda: float = Field(default=0.03, ge=0.0)


class KimiChatRequest(BaseModel):
    api_key: str | None = None
    model: str = Field(default="kimi-k2.6", min_length=1)
    endpoint: str = Field(default="https://api.moonshot.cn/v1/chat/completions", min_length=1)
    messages: list[dict[str, str]]
    thinking_mode: str = Field(default="fast")
    max_completion_tokens: int = Field(default=8192, ge=1, le=8192)
    timeout_seconds: float = Field(default=180.0, gt=0, le=300)


class StyleTransferRequest(BaseModel):
    model_path: str
    tracks_path: str
    source_track_id: str
    style_track_id: str
    out_name: str = Field(default="style_transfer.npz")
    target_culture: str | None = None
    alpha: float = Field(default=1.0, ge=0.0, le=2.0)
    k: int = Field(default=10, ge=1, le=200)
    prefer_cuda: bool = Field(default=False)


class WaveStyleTransferRequest(BaseModel):
    source_audio_path: str
    style_audio_path: str
    out_name: str = Field(default="style_transfer_wave.wav")
    alpha: float = Field(default=0.7, ge=0.0, le=1.5)
    target_sr: int = Field(default=24000, ge=8000, le=96000)
    n_fft: int = Field(default=1024, ge=128, le=8192)
    hop_length: int = Field(default=256, ge=32, le=4096)
    win_length: int = Field(default=1024, ge=128, le=8192)
    max_seconds: float | None = Field(default=12.0, gt=0)
    peak_norm: float = Field(default=0.98, gt=0.1, le=1.0)


class OntologyConceptCreateRequest(BaseModel):
    name: str
    description: str = ""
    parent_id: str | None = None
    aliases: list[str] = Field(default_factory=list)


class OntologyRelationCreateRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    weight: float = Field(default=1.0)


class OntologyAnnotationCreateRequest(BaseModel):
    track_id: str
    concept_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="expert")
    rationale: str = ""


class OntologySuggestRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class OntologyExportConstraintsRequest(BaseModel):
    out_name: str = Field(default="ontology_constraints.jsonl")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_pairs_per_concept: int = Field(default=200, ge=1, le=100000)


class PalRequest(BaseModel):
    model_path: str
    tracks_path: str
    out_name: str = Field(default="pal_tasks.jsonl")
    n: int = Field(default=100, ge=1, le=2000)
    prefer_cuda: bool = Field(default=False)


class PrototypeRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=3, max_length=200)
    role: str = Field(default="听众", min_length=1, max_length=40)


class PrototypeAnalyzeRequest(BaseModel):
    upload_id: str
    mode: str = Field(default="bridge")
    lens: str = Field(default="rhythm")
    engine: str = Field(default="echo")
    top_k: int = Field(default=3, ge=1, le=6)


class PrototypeFeedbackRequest(BaseModel):
    track: str = Field(min_length=1, max_length=240)
    recommendation_id: str | None = Field(default=None, max_length=120)
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=2000)
    profile_id: str | None = Field(default=None, max_length=120)
