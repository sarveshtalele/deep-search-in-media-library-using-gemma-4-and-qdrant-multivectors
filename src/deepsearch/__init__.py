"""Deep Search in Media Library Using Gemma 4 and Qdrant Multivectors.

A local, privacy-first multimodal media search engine that ingests video,
audio and text, indexes them in Qdrant named/multi-vectors, and serves
cross-modal natural-language search plus temporal RAG — all powered by a
locally hosted Gemma 4 model via Ollama.
"""

__version__ = "0.1.0"

from deepsearch.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
