"""LLM judge with structured JSON output for emoji toxicity classification.

The model emits a calibrated *toxicity_score* in [0, 1] (0 = clearly SAFE, 1 =
clearly TOXIC). The final TOXIC / SAFE / UNCERTAIN verdict is derived from this
score by the confidence gate in pipeline.py — the model does not emit a verdict
string, eliminating the old calibration pathology where "confidence" meant
"confidence in verdict" (symmetric, AUROC-uninterpretable).
"""

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
    toxicity_score: float = Field(
        description="Probability the message is toxic. 0.0 = clearly SAFE, "
        "1.0 = clearly TOXIC, 0.5 = genuinely uncertain."
    )
    reasoning: str = Field(description="Brief explanation of the score")
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
4. Each retrieved entry has a Relevance score (0-1). Discount entries with low relevance (< 0.5) — they may not match the emoji in question.

Output structured JSON with:
- toxicity_score: a calibrated probability the message is toxic.
    * 0.0-0.2: clearly SAFE (literal usage, benign context)
    * 0.3-0.6: genuinely UNCERTAIN (ambiguous context, mixed signals)
    * 0.7-1.0: clearly TOXIC (coded language, harmful intent)
  The score must be monotone with toxicity — higher = more toxic. Do NOT encode
  "confidence in my answer" here; encode "probability the message is toxic".
- reasoning: brief explanation
- risk_category: Hate Speech, Sexual, Political, Drug, Safe, or Unclear
- emoji_analysis: per-emoji breakdown with emoji, interpretation, and risk level"""

HUMAN_TEMPLATE = """Context: {context_text}
Message: {message}

Analyze the emoji usage and return the structured JSON."""


@lru_cache(maxsize=8)
def _get_chain(seed: int | None = None):
    """Build the prompt → structured-LLM chain. Cached per seed for multi-seed eval."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        seed=seed,
    )
    return prompt | llm.with_structured_output(ClassificationResult)


def classify(
    message: str,
    context_text: str,
    retrieved_knowledge: str,
    seed: int | None = None,
) -> ClassificationResult:
    """Run the LLM judge on a message with context and retrieved knowledge."""
    return _get_chain(seed).invoke({
        "context": retrieved_knowledge,
        "context_text": context_text,
        "message": message,
    })
