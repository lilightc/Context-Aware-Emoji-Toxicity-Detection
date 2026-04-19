"""Tools exposed to the toxicity detection agent.

Each tool is a LangChain @tool-decorated function with a typed schema. The agent
reasons about which tools to call based on the message and context; the fixed
workflow (``mode="workflow"``) bypasses this and always calls retrieval.

``submit_verdict`` is terminal: when the agent calls it, the loop exits. The
agent emits a toxicity_score (0=SAFE, 1=TOXIC), not a verdict string — the final
TOXIC/SAFE/UNCERTAIN label is derived from the score by the confidence gate.
"""

from __future__ import annotations

from langchain_core.tools import tool

from emoji_toxicity.detector.classifier import ClassificationResult, EmojiAnalysis
from emoji_toxicity.detector.retriever import retrieve
from emoji_toxicity.utils import cldr_name, format_retrieved_docs


@tool
def lookup_emoji_knowledge(emoji: str) -> str:
    """Retrieve slang, risk category, and context signals for a specific emoji
    from the knowledge base. Use when you want information about a single emoji
    whose meaning may be coded/slang.

    Args:
        emoji: A single emoji character, e.g. "🌽"
    """
    result = retrieve(emoji, k=3)
    text, _ = format_retrieved_docs(result.documents, prefix=f"Knowledge for {emoji}")
    return text


@tool
def search_similar_cases(query: str) -> str:
    """Free-form semantic search against the knowledge base. Use for multi-emoji
    combinations, slang phrases, or when you want context-aware matches that
    single-emoji lookup may miss.

    Args:
        query: A free-form text query (may contain emoji and words).
    """
    result = retrieve(query, k=3)
    text, _ = format_retrieved_docs(result.documents, prefix=f"Similar cases for '{query}'")
    return text


@tool
def get_cldr_name(emoji: str) -> str:
    """Return the official Unicode CLDR short name for an emoji (e.g. 'ear of
    corn'). Cheap local lookup — prefer this over knowledge-base search when you
    only need the literal meaning.

    Args:
        emoji: A single emoji character.
    """
    name = cldr_name(emoji)
    return name or f"No CLDR name found for {emoji}"


@tool
def submit_verdict(
    toxicity_score: float,
    reasoning: str,
    risk_category: str,
    emoji_analysis: list[dict],
) -> str:
    """TERMINAL. Submit the final toxicity score. Calling this ends the agent loop.

    Args:
        toxicity_score: Probability the message is toxic, in [0, 1].
            0.0-0.2 = clearly SAFE, 0.3-0.6 = UNCERTAIN, 0.7-1.0 = clearly TOXIC.
            MUST be monotone with toxicity — higher = more toxic.
        reasoning: Brief explanation of the score.
        risk_category: One of "Hate Speech", "Sexual", "Political", "Drug", "Safe", "Unclear".
        emoji_analysis: List of {emoji, interpretation, risk} dicts — one per emoji in the message.
    """
    return "Verdict submitted."


def as_classification_result(args: dict) -> ClassificationResult:
    """Reconstruct a ClassificationResult from submit_verdict's tool arguments."""
    return ClassificationResult(
        toxicity_score=float(args["toxicity_score"]),
        reasoning=args["reasoning"],
        risk_category=args["risk_category"],
        emoji_analysis=[EmojiAnalysis(**ea) for ea in args.get("emoji_analysis", [])],
    )


AGENT_TOOLS = [lookup_emoji_knowledge, search_similar_cases, get_cldr_name, submit_verdict]
TERMINAL_TOOL_NAME = "submit_verdict"
