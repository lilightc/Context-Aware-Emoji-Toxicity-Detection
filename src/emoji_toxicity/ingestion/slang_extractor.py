"""LLM-assisted emoji slang extraction from raw social-media text.

Given a batch of social-media posts containing emoji, uses GPT to identify
which emoji are being used as coded language and extract structured KB entries.
"""

from __future__ import annotations

import json
from functools import lru_cache

from openai import OpenAI

from emoji_toxicity.config import settings
from emoji_toxicity.utils import extract_emojis, cldr_name


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def extract_slang_candidates(
    posts: list[dict],
    model: str | None = None,
) -> list[dict]:
    """Extract emoji slang candidates from raw social-media posts.

    Args:
        posts: List of {"text": str, "source": str, "url": str (optional)}.
        model: LLM model override. Defaults to settings.llm_model.

    Returns:
        List of candidate KB entries with extraction metadata:
        {symbol, literal_meaning, slang_meaning, risk_category, toxic_signals,
         benign_signals, confidence, source_posts, extraction_model}
    """
    if not posts:
        return []

    # Batch posts into a single prompt (up to 20 per batch to stay in context)
    candidates = []
    for i in range(0, len(posts), 20):
        batch = posts[i : i + 20]
        batch_candidates = _extract_batch(batch, model or settings.llm_model)
        candidates.extend(batch_candidates)

    return candidates


def _extract_batch(posts: list[dict], model: str) -> list[dict]:
    """Extract slang from a batch of posts."""
    posts_text = "\n".join(
        f"[{i + 1}] (source: {p.get('source', '?')}) {p['text']}"
        for i, p in enumerate(posts)
    )

    prompt = f"""You are an emoji slang analyst. Analyze these social-media posts and identify
any emoji being used as CODED LANGUAGE (slang, dog whistles, euphemisms) rather than literally.

Posts:
{posts_text}

For each emoji you identify as coded, return a JSON object. If an emoji is used literally,
skip it. Return a JSON array (empty array if no coded usage found).

Each object must have:
{{
    "symbol": "the emoji character",
    "slang_meaning": "what the coded meaning is",
    "risk_category": "one of: Hate Speech, Sexual, Political, Drug, Safe",
    "toxic_signals": ["5 context keywords suggesting coded usage"],
    "benign_signals": ["5 context keywords suggesting literal usage"],
    "confidence": 0.0-1.0 (how confident are you that this is ESTABLISHED SLANG, not a one-off joke?
        1.0 = widely documented, appears across multiple communities
        0.7 = likely real slang, seen in several posts
        0.4 = might be slang, only seen in one context
        0.1 = probably a one-off joke or literal misread),
    "evidence_posts": [list of post numbers where this usage appears]
}}

Return ONLY a valid JSON array. No markdown."""

    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON arrays. No markdown."},
            {"role": "user", "content": prompt},
        ],
    )

    try:
        raw = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw, list):
        return []

    # Enrich with metadata
    for entry in raw:
        entry["literal_meaning"] = cldr_name(entry.get("symbol", ""))
        entry["extraction_model"] = model
        entry["source_posts"] = [
            posts[idx - 1] for idx in entry.get("evidence_posts", [])
            if 0 < idx <= len(posts)
        ]

    return raw
