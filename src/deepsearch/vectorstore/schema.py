"""Collection schema, named-vector layout and the Fragment data model.

Retrieval unit = **fragment** (one keyframe, one audio chunk, or one text
description). Each fragment is a Qdrant point that populates exactly one of the
three named vector spaces and carries a payload with the precise temporal
offsets needed for second-accurate playback. This is what makes timestamp-level
RAG possible — an asset-level multivector would score the whole file but lose
the pointer back to the matching second.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

# The three single-vector named spaces (fragment-level, temporal targeting).
NAMED_VECTORS = ("text_descriptions", "video_frames", "audio_chunks")

# Asset-level MaxSim multivector spaces (ColBERT-style arrays per asset). One
# point per asset holds every keyframe / audio-chunk vector, scored with MaxSim.
# Used for fast asset-level recall; the fragment points above keep the exact
# timestamps. See docs/ARCHITECTURE_REVIEW.md §3 & §5.
MULTIVECTOR_VECTORS = ("frames_maxsim", "audio_maxsim")
MULTIVECTOR_FOR_MODALITY = {
    "video_frames": "frames_maxsim",
    "audio_chunks": "audio_maxsim",
}


class Modality(str, Enum):
    TEXT = "text_descriptions"
    VIDEO = "video_frames"
    AUDIO = "audio_chunks"


MODALITIES = tuple(m.value for m in Modality)


@dataclass
class Fragment:
    """A single indexable media fragment."""

    asset_id: str               # stable id shared by all fragments of one file
    asset_name: str             # display name (file stem)
    file_path: str              # absolute path to the source media on disk
    modality: Modality
    text: str                   # description / transcript / caption (embedded)
    start_s: float = 0.0        # fragment start offset within the asset
    end_s: float = 0.0          # fragment end offset within the asset
    duration_s: float = 0.0     # total duration of the parent asset
    category: str = "uncategorized"
    thumbnail_path: str | None = None  # optional keyframe image path for the UI
    ingested_at: int = field(default_factory=lambda: int(time.time()))
    point_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def payload(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "file_path": self.file_path,
            "modality": self.modality.value,
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "category": self.category,
            "thumbnail_path": self.thumbnail_path,
            "ingested_at": self.ingested_at,
        }


# Payload fields that get indexed for fast pre-filtering.
KEYWORD_INDEXES = ("asset_id", "modality", "category")
INTEGER_INDEXES = ("ingested_at",)
