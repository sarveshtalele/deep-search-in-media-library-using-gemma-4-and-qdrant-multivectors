"""Typed, layered configuration.

Resolution order (lowest to highest precedence):
    1. ``config/default.yaml``
    2. environment variables prefixed ``DEEPSEARCH_`` (and ``.env``)

Everything in the codebase reads settings through :func:`get_settings`, which
is cached so the YAML/env are parsed exactly once per process.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class ModelSettings(BaseModel):
    ollama_host: str = "http://localhost:11434"
    gemma_model: str = "gemma4:e4b"
    embed_model: str = "nomic-embed-text"
    embed_strategy: Literal["describe_then_embed", "native"] = "describe_then_embed"
    embed_dim: int = 768
    request_timeout_s: int = 120
    # Latency controls: cap generated tokens and keep the model resident in VRAM
    # between calls so it never reloads. Big wins on local Apple Silicon.
    num_predict: int = 224          # max tokens per description (room for accurate detail)
    gen_num_predict: int = 256      # max tokens for intent/rerank generation
    chat_num_predict: int = 512     # max tokens for RAG chat answers (counting/listing)
    num_ctx: int = 8192             # context window for RAG chat (fit full timeline)
    keep_alive: str = "30m"
    asr_model: str = "base"         # faster-whisper model: tiny|base|small (speed/quality)


class QuantizationSettings(BaseModel):
    enabled: bool = True
    type: Literal["scalar", "product", "none"] = "scalar"
    always_ram: bool = True


class QdrantSettings(BaseModel):
    url: str = ""
    path: str = "./qdrant_storage"
    collection: str = "media_library"
    distance: Literal["Cosine", "Dot", "Euclid"] = "Cosine"
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)


class IngestionSettings(BaseModel):
    # Audio transcription backend: "whisper" (faster-whisper, one fast pass — much
    # faster on long media) or "gemma" (per-chunk Gemma audio calls).
    audio_backend: Literal["whisper", "gemma"] = "whisper"
    audio_chunk_seconds: int = 30
    audio_overlap_seconds: int = 5
    audio_denoise: bool = True       # apply FFmpeg afftdn noise reduction
    max_audio_chunks: int = 180      # cap chunks on very long media (grows window)
    scene_threshold: float = 27.0
    # Above this duration, skip full scene-detection (which decodes the entire
    # video) and sample keyframes uniformly — we only keep `max_keyframes` anyway.
    scene_detect_max_duration: int = 900  # seconds (15 min)
    max_keyframes: int = 12          # fewer Gemma vision calls = much faster ingest
    frame_resize: int = 896          # higher res -> more accurate counting/detail
    batch_size: int = 16
    # Gemma describe/transcribe calls run concurrently — Ollama serves them in
    # parallel, giving ~3x throughput on long media (the main ingest cost).
    ingest_workers: int = 4


class RetrievalSettings(BaseModel):
    top_k: int = 10
    prefetch_limit: int = 30
    fusion: Literal["rrf", "dbsf", "weighted"] = "rrf"
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "text_descriptions": 0.5,
            "video_frames": 0.3,
            "audio_chunks": 0.2,
        }
    )
    # Both default OFF for fast local search: an LLM rerank/intent call costs
    # 10-100s on e4b. Heuristic intent is instant; RRF fusion is already strong.
    # Enable per-request when quality matters more than latency.
    intent_llm: bool = False
    rerank: bool = False
    rerank_top_n: int = 8


class SafetySettings(BaseModel):
    enabled: bool = True
    max_query_chars: int = 2000


class RuntimeSettings(BaseModel):
    use_stub: bool = False
    log_level: str = "INFO"


class Settings(BaseSettings):
    """Top-level application settings."""

    models: ModelSettings = Field(default_factory=ModelSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    model_config = SettingsConfigDict(
        env_prefix="DEEPSEARCH_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    @property
    def qdrant_storage_path(self) -> Path:
        return (PROJECT_ROOT / self.qdrant.path).resolve()


# ----------------------------------------------------------------------------
# Flat env overrides
# ----------------------------------------------------------------------------
# pydantic-settings nested env vars are verbose. We also honour a small set of
# friendly flat names documented in .env.example.
_FLAT_ENV: dict[str, tuple[str, str]] = {
    "DEEPSEARCH_OLLAMA_HOST": ("models", "ollama_host"),
    "DEEPSEARCH_GEMMA_MODEL": ("models", "gemma_model"),
    "DEEPSEARCH_EMBED_MODEL": ("models", "embed_model"),
    "DEEPSEARCH_EMBED_STRATEGY": ("models", "embed_strategy"),
    "DEEPSEARCH_ASR_MODEL": ("models", "asr_model"),
    "DEEPSEARCH_AUDIO_BACKEND": ("ingestion", "audio_backend"),
    "DEEPSEARCH_QDRANT_URL": ("qdrant", "url"),
    "DEEPSEARCH_QDRANT_PATH": ("qdrant", "path"),
    "DEEPSEARCH_COLLECTION": ("qdrant", "collection"),
    "DEEPSEARCH_AUDIO_CHUNK_SECONDS": ("ingestion", "audio_chunk_seconds"),
    "DEEPSEARCH_AUDIO_OVERLAP_SECONDS": ("ingestion", "audio_overlap_seconds"),
    "DEEPSEARCH_SCENE_THRESHOLD": ("ingestion", "scene_threshold"),
    "DEEPSEARCH_MAX_KEYFRAMES": ("ingestion", "max_keyframes"),
    "DEEPSEARCH_MAX_AUDIO_CHUNKS": ("ingestion", "max_audio_chunks"),
    "DEEPSEARCH_INGEST_WORKERS": ("ingestion", "ingest_workers"),
    "DEEPSEARCH_TOP_K": ("retrieval", "top_k"),
    "DEEPSEARCH_RERANK": ("retrieval", "rerank"),
    "DEEPSEARCH_USE_STUB": ("runtime", "use_stub"),
    "DEEPSEARCH_LOG_LEVEL": ("runtime", "log_level"),
}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _coerce(raw: str) -> object:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_flat_env(data: dict) -> dict:
    for env_name, (section, key) in _FLAT_ENV.items():
        if env_name in os.environ:
            data.setdefault(section, {})[key] = _coerce(os.environ[env_name])
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the resolved application settings."""
    data = _load_yaml(DEFAULT_CONFIG)
    data = _apply_flat_env(data)
    return Settings(**data)
