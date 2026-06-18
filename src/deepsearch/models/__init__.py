"""Model access layer.

Everything the application needs from Gemma 4 (embeddings, multimodal
description, generation, reranking) is funnelled through a single
:class:`~deepsearch.models.gemma.GemmaModelClient`, so the rest of the
codebase never talks to Ollama directly.
"""

from deepsearch.models.gemma import GemmaModelClient, get_model_client
from deepsearch.models.safety import SafetyVerdict, check_query, sanitize_generated

__all__ = [
    "GemmaModelClient",
    "get_model_client",
    "SafetyVerdict",
    "check_query",
    "sanitize_generated",
]
