"""Shared utilities used across ingestion, vectorstore, and detector modules."""

from __future__ import annotations

import emoji


def extract_emojis(text: str) -> list[str]:
    """Return all emoji characters present in text, in order."""
    return [ch for ch in text if ch in emoji.EMOJI_DATA]


def cldr_name(symbol: str) -> str:
    """Return the human-readable CLDR short name for an emoji (e.g. 'ear of corn')."""
    data = emoji.EMOJI_DATA.get(symbol, {})
    return data.get("en", "").strip(":").replace("_", " ")


def make_vec_id(symbol: str) -> str:
    """Build a deterministic vector ID for a (possibly multi-codepoint) emoji symbol."""
    return "vec_" + "-".join(f"{ord(ch):x}" for ch in symbol)
