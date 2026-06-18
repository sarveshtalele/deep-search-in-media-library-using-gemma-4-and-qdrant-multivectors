"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    stub_mode: bool
    ollama_host: str
    gemma_model: str
    embed_model: str
    ollama_reachable: bool
    gemma_present: bool
    embed_present: bool
    models: list[str] = []
    detail: str = ""


class UploadResponse(BaseModel):
    filename: str
    kind: str          # video | audio | image | text | unknown
    size_bytes: int
    supported: bool


class IngestRequest(BaseModel):
    filename: str
    category: str = "uncategorized"


class SearchRequest(BaseModel):
    query: str
    categories: list[str] | None = None
    modalities: list[str] | None = None
    weights: dict[str, float] | None = None
    asset_recall: bool = False


class Hit(BaseModel):
    point_id: str
    asset_id: str
    asset_name: str
    file_path: str
    media_url: str
    thumbnail_url: str | None
    modality: str
    text: str
    start_s: float
    end_s: float
    timestamp_label: str
    score: float
    contributions: dict[str, float] = {}


class SearchResponse(BaseModel):
    query: str
    intent: str
    weights: dict[str, float]
    fusion: str
    blocked: bool
    message: str
    hits: list[Hit]


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


class AskSource(BaseModel):
    asset_id: str
    asset_name: str
    media_url: str
    modality: str
    start_s: float
    timestamp_label: str


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource] = []


class StatsResponse(BaseModel):
    assets: int
    fragments: int
    by_modality: dict[str, int]


class AssetOut(BaseModel):
    asset_id: str
    asset_name: str
    file_path: str
    media_url: str
    modalities: list[str]
    has_audio: bool


class DeleteResponse(BaseModel):
    ok: bool
    asset_id: str
