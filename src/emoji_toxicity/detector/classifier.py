"""LLM judge with structured JSON output for emoji toxicity classification."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from emoji_toxicity.config import settings


class EmojiAnalysis(BaseModel):
    """Analysis of a single emoji in context."""
    emoji: str = Field(description="The emoji character")
    interpretation: str = Field(description="How the emoji is being used: literal or slang")
    risk: str = Field(description="Risk level: high, medium, low, none")


class ClassificationResult(BaseModel):
    """Structured output from the LLM judge."""
    verdict: str = Field(description="One of: TOXIC, SAFE, UNCERTAIN")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the verdict")
    risk_category: str = Field(
        description="Category: Hate Speech, Sexual, Political, Drug, Safe, Unclear"
    )
    emoji_analysis: list[EmojiAnalysis] = Field(description="Per-emoji breakdown")


SYSTEM_PROMPT = """You are a Content Safety Expert specializing in emoji-based coded language.

Analyze the relationship between the message, its context, and the retrieved knowledge about emoji meanings.

[RETRIEVED KNOWLEDGE]
{context}

[TASK]
1. For each emoji, determine if it is used LITERALLY (benign) or as CODED LANGUAGE (slang/dog whistle).
2. Consider the surrounding context — the same emoji can be toxic or safe depending on usage.
3. Use the retrieved knowledge to inform your decision, but apply judgment.

Output your analysis as structured JSON with:
- verdict: TOXIC, SAFE, or UNCERTAIN (use UNCERTAIN when context is ambiguous)
- confidence: 0.0 to 1.0
- reasoning: brief explanation
- risk_category: Hate Speech, Sexual, Political, Drug, Safe, or Unclear
- emoji_analysis: per-emoji breakdown with emoji, interpretation, and risk level"""

HUMAN_TEMPLATE = """Context: {context_text}
Message: {message}

Analyze the emoji usage in this message."""


@lru_cache(maxsize=1)
def _get_chain():
    """Build the prompt → structured-LLM chain once and cache it."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )
    return prompt | llm.with_structured_output(ClassificationResult)


def classify(
    message: str,
    context_text: str,
    retrieved_knowledge: str,
) -> ClassificationResult:
    """Run the LLM judge on a message with context and retrieved knowledge."""
    return _get_chain().invoke({
        "context": retrieved_knowledge,
        "context_text": context_text,
        "message": message,
    })
