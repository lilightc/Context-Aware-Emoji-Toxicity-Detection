"""Lightweight retrieval gate — decides whether a message needs KB lookup.

This replaces the agent's GPT-5-based tool selection with a fast heuristic
classifier. If the gate says "no retrieval needed", the workflow skips the
Pinecone query and runs the LLM directly (like raw_llm but with the structured
toxicity_score prompt).

The gate checks:
1. Whether any emoji in the message are in a known "potentially coded" set
2. Whether the context contains slang trigger keywords
3. Whether the message is short enough to be an emoji-only or emoji-dominant message

If none of these fire, the message is likely literal and retrieval adds noise.
"""

from __future__ import annotations

from emoji_toxicity.utils import extract_emojis

# Emoji known to carry coded/slang meanings — union of our KB's non-Safe entries
# and commonly documented dog-whistle emoji. This set is the "retrieval trigger".
CODED_EMOJI: set[str] = {
    # Sexual
    "🌽", "🍑", "🍆", "🍒", "🥜", "🦴", "💦", "👉", "👌",
    # Hate speech
    "🐵", "🐒", "🦍", "🐷",
    # Political
    "👌🏻", "🐸", "🥛",
    # Drug
    "❄️", "🍄", "🌿", "💊", "🍃",
    # Bullying / ambiguous
    "🤡", "🐍", "💀", "☠️", "🗑️", "🔫", "🖕", "🔪",
    # Multi-codepoint variants
    "👌🏻", "👌🏼", "👌🏽", "👌🏾", "👌🏿",
}

# Context keywords that suggest the emoji may be coded even if the emoji itself
# is common (e.g., "🍑 is unreal" + "beach" vs "🍑 is unreal" + "brewery")
SLANG_CONTEXT_TRIGGERS: set[str] = {
    "link in bio", "dm me", "hmu", "exclusive content", "onlyfans", "plug me",
    "if you know", "iykyk", "no cap", "fr fr", "based",
    "hit me up", "hook up", "send me your",
}


def needs_retrieval(text: str, context: str = "") -> bool:
    """Return True if the message likely needs KB retrieval.

    Fast heuristic — no API calls, no vector search. Designed to be
    high-recall (err on the side of retrieving) so we don't miss coded usage.
    """
    emojis = extract_emojis(text)

    # No emoji → nothing to look up
    if not emojis:
        return False

    # Any known-coded emoji present → retrieve
    if any(e in CODED_EMOJI for e in emojis):
        return True

    # Context contains slang triggers → retrieve even for "safe" emoji
    combined = (text + " " + context).lower()
    if any(trigger in combined for trigger in SLANG_CONTEXT_TRIGGERS):
        return True

    # Emoji-dominant message (>50% emoji by character count) → likely coded
    emoji_ratio = len(emojis) / max(len(text.replace(" ", "")), 1)
    if emoji_ratio > 0.3:
        return True

    return False
