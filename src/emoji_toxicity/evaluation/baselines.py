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
    """Same score convention as ClassificationResult: 0=SAFE, 1=TOXIC."""
    toxicity_score: float = Field(
        description="Probability the message is toxic. 0.0=clearly SAFE, 1.0=clearly TOXIC."
    )


def keyword_baseline(text: str, context: str = "") -> BaselineResult:
    """Blacklist: known toxic emoji present → high toxicity score."""
    if any(ch in TOXIC_EMOJI_BLACKLIST for ch in extract_emojis(text)):
        return BaselineResult(toxicity_score=0.8)
    return BaselineResult(toxicity_score=0.2)


@lru_cache(maxsize=8)
def _get_raw_llm(seed: int | None = None):
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
        seed=seed,
    )
    return llm.with_structured_output(BaselineResult)


def raw_llm_baseline(
    text: str, context: str = "", seed: int | None = None
) -> BaselineResult:
    """Raw LLM classification without RAG — to show RAG adds value."""
    prompt = f"""Rate the toxicity of this message on a 0-to-1 scale.
0.0 = clearly SAFE (literal/benign usage)
1.0 = clearly TOXIC (coded language, harassment, hate speech)
0.5 = genuinely ambiguous

Context: {context or 'None'}
Message: {text}

Return toxicity_score only. The score must be monotone — higher means more toxic."""
    return _get_raw_llm(seed).invoke(prompt)
