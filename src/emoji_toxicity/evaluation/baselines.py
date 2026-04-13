"""Baseline classifiers for comparison: keyword blacklist, raw LLM (no RAG)."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from emoji_toxicity.config import settings
from emoji_toxicity.utils import extract_emojis

TOXIC_EMOJI_BLACKLIST = {
    "🍆", "🍑", "🌽", "💀", "🐵", "🐒", "🤡", "🐍", "👌🏻",
    "🔫", "🖕", "💊", "🍄", "❄️", "🌿", "🥜", "🦍", "🐷",
}


class BaselineResult(BaseModel):
    verdict: str = Field(description="TOXIC or SAFE")
    confidence: float = Field(description="0.0 to 1.0")


def keyword_baseline(text: str, context: str = "") -> BaselineResult:
    """Simple blacklist: if any emoji in text is in the toxic set → TOXIC."""
    if any(ch in TOXIC_EMOJI_BLACKLIST for ch in extract_emojis(text)):
        return BaselineResult(verdict="TOXIC", confidence=0.8)
    return BaselineResult(verdict="SAFE", confidence=0.8)


@lru_cache(maxsize=1)
def _get_raw_llm():
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    return llm.with_structured_output(BaselineResult)


def raw_llm_baseline(text: str, context: str = "") -> BaselineResult:
    """Raw LLM classification without RAG — to show RAG adds value."""
    prompt = f"""Classify whether this message contains toxic emoji usage.
Context: {context or 'None'}
Message: {text}

Consider whether emoji are used as coded language, dog whistles, or harassment."""
    return _get_raw_llm().invoke(prompt)
