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


def verdict_from_score(
    score: float, toxic_threshold: float = 0.7, safe_threshold: float = 0.3
) -> str:
    """Derive TOXIC/SAFE/UNCERTAIN verdict from a monotone toxicity score."""
    if score >= toxic_threshold:
        return "TOXIC"
    if score <= safe_threshold:
        return "SAFE"
    return "UNCERTAIN"


def format_retrieved_docs(
    documents, scores: list[float] | None = None, prefix: str = ""
) -> tuple[str, list[str]]:
    """Format retrieval docs into (text_for_llm, citations_list).

    When ``scores`` are provided (cosine similarity from Pinecone), each entry
    includes a relevance score so the LLM can discount low-similarity retrievals.
    """
    if not documents:
        return f"{prefix or 'Retrieved knowledge'}: no relevant entries.", []
    parts, citations = [], []
    for i, doc in enumerate(documents, 1):
        m = doc.metadata
        symbol = m.get("symbol", "?")
        score_str = f"  Relevance: {scores[i - 1]:.3f}\n" if scores else ""
        parts.append(
            f"[{i}] {symbol}\n"
            f"{score_str}"
            f"  Slang meaning: {m.get('slang_meaning', 'N/A')}\n"
            f"  Risk category: {m.get('risk_category', 'Unknown')}\n"
            f"  Toxic signals: {m.get('toxic_signals', 'N/A')}\n"
            f"  Benign signals: {m.get('benign_signals', 'N/A')}"
        )
        citations.append(f"{symbol}: {m.get('slang_meaning', 'N/A')}")
    return (f"{prefix}:\n" if prefix else "") + "\n\n".join(parts), citations
