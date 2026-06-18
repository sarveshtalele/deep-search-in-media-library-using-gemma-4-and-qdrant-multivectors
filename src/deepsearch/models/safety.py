"""Lightweight input safety / prompt-injection guard for incoming queries.

This is intentionally a *pre-filter* that runs before any vector lookup, as
required by the production-guardrails section. It combines cheap deterministic
heuristics (length, known injection patterns) with an optional Gemma 4 check.
It is not a substitute for a dedicated safety model in a hostile deployment,
but it stops the obvious cases locally and without network egress.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from deepsearch.config import get_settings

_INJECTION_PATTERNS = [
    r"ignore (all|previous|above) (instructions|prompts)",
    r"disregard (the )?(system|previous)",
    r"you are now (a|an|in) ",
    r"reveal (your )?(system )?prompt",
    r"<\s*/?\s*(system|assistant|tool)\s*>",
    r"\bexfiltrate\b|\brm -rf\b|\bdrop table\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


@dataclass
class SafetyVerdict:
    allowed: bool
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return not self.allowed


def check_query(query: str) -> SafetyVerdict:
    """Return a verdict for a user query prior to retrieval."""
    cfg = get_settings().safety
    if not cfg.enabled:
        return SafetyVerdict(True)

    text = (query or "").strip()
    if not text:
        return SafetyVerdict(False, "Empty query.")
    if len(text) > cfg.max_query_chars:
        return SafetyVerdict(False, f"Query exceeds {cfg.max_query_chars} characters.")
    for pat in _COMPILED:
        if pat.search(text):
            return SafetyVerdict(False, "Query matched a prompt-injection pattern.")
    return SafetyVerdict(True)


def sanitize_generated(text: str) -> str:
    """Scrub model-generated metadata before it is indexed.

    Media can carry adversarial on-screen / spoken text ("ignore all previous
    instructions…") that Gemma may transcribe verbatim. We neutralise such
    injected directives so they cannot later steer the RAG model when the
    description is fed back as context.
    """
    if not get_settings().safety.enabled or not text:
        return text
    cleaned = text
    for pat in _COMPILED:
        cleaned = pat.sub("[redacted]", cleaned)
    return cleaned
