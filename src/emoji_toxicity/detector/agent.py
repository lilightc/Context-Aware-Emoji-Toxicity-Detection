"""Tool-calling agent for context-aware emoji toxicity detection.

Unlike the fixed workflow (retrieve → classify → gate), the agent decides:
- Whether to retrieve at all (literal use in obvious context may not need retrieval)
- Which emoji or phrase to look up
- Which tool to use (targeted emoji lookup vs. free-form semantic search vs.
  cheap CLDR lookup)

The agent terminates by calling the ``submit_verdict`` tool with a
ClassificationResult. A max-iteration cap bounds cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from emoji_toxicity.config import settings
from emoji_toxicity.detector.classifier import ClassificationResult
from emoji_toxicity.detector.tools import (
    AGENT_TOOLS,
    TERMINAL_TOOL_NAME,
    as_classification_result,
)


AGENT_SYSTEM_PROMPT = """You are a Content Safety Expert analyzing emoji usage for toxicity.

The same emoji can be benign or harmful depending on context (🌽 = corn OR porn; \
🐵 = monkey OR racial slur; 🍑 = peach OR sexual). Your job is to rate the \
toxicity of the message on a 0-to-1 scale.

You have tools to gather information:
- get_cldr_name(emoji): cheap local lookup of an emoji's literal name
- lookup_emoji_knowledge(emoji): retrieve slang/risk info for one emoji from the KB
- search_similar_cases(query): free-form semantic search — useful for multi-emoji \
phrases, slang combinations, or non-obvious patterns
- submit_verdict(toxicity_score, reasoning, risk_category, emoji_analysis): TERMINAL. \
Call this exactly once when you are ready with the final score. This ends your turn.

Scoring guidance:
- toxicity_score ∈ [0, 1], monotone in toxicity — higher means more toxic.
  * 0.0-0.2 = clearly SAFE (literal usage, benign context)
  * 0.3-0.6 = genuinely UNCERTAIN (ambiguous signals, unclear intent)
  * 0.7-1.0 = clearly TOXIC (coded language, harmful intent)
- Do NOT encode "how confident I am in my answer" here. Encode "probability the \
message is toxic". A confidently-SAFE message should score NEAR 0.0, not near 1.0.

Tool-use guidance:
- If the message is clearly literal (e.g. emoji in obvious food/travel/weather \
context and the surrounding text supports the literal reading), you MAY skip \
retrieval entirely and submit_verdict directly.
- If an emoji may carry coded/slang meaning OR the context is ambiguous, call \
lookup_emoji_knowledge for it.
- For multi-emoji combinations or short coded phrases, prefer search_similar_cases.
- Call at most 2-3 information tools before submitting the verdict.

Always end by calling submit_verdict. Do not emit plain-text responses."""


@dataclass
class AgentTrace:
    """Record of tool calls made during a single agent run (for debugging/analysis)."""
    iterations: int
    tool_calls: list[dict]  # list of {"name": ..., "args": ..., "result": ...}
    terminated_normally: bool


@dataclass
class AgentResult:
    classification: ClassificationResult
    trace: AgentTrace


@lru_cache(maxsize=8)
def _get_llm_with_tools(seed: int | None = None):
    """Build and cache the GPT-5 model bound to the agent's tools."""
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        seed=seed,
    )
    return llm.bind_tools(AGENT_TOOLS)


_TOOLS_BY_NAME = {t.name: t for t in AGENT_TOOLS}


def run_agent(
    message: str,
    context: str = "",
    seed: int | None = None,
    max_iterations: int = 4,
) -> AgentResult:
    """Run the tool-calling agent on a message.

    Returns the final ClassificationResult plus a trace of tool calls. If the
    agent fails to terminate via submit_verdict within ``max_iterations``,
    returns an UNCERTAIN verdict flagged in the trace.
    """
    tools_map = _TOOLS_BY_NAME
    llm = _get_llm_with_tools(seed)

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context: {context or 'No additional context provided.'}\n"
                f"Message: {message}\n\n"
                "Analyze the emoji usage and submit your verdict."
            )
        ),
    ]

    trace_calls: list[dict] = []

    for iteration in range(max_iterations):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            # Nudge the model to use tools rather than emitting plain text
            messages.append(HumanMessage(content="Please finish by calling submit_verdict."))
            continue

        for tc in tool_calls:
            name = tc["name"]
            args = tc["args"]
            tool_call_id = tc["id"]

            if name == TERMINAL_TOOL_NAME:
                trace_calls.append({"name": name, "args": args, "result": "<terminal>"})
                classification = as_classification_result(args)
                return AgentResult(
                    classification=classification,
                    trace=AgentTrace(
                        iterations=iteration + 1,
                        tool_calls=trace_calls,
                        terminated_normally=True,
                    ),
                )

            tool_fn = tools_map.get(name)
            if tool_fn is None:
                result_text = f"Unknown tool: {name}"
            else:
                try:
                    result_text = tool_fn.invoke(args)
                except Exception as e:
                    result_text = f"Tool error: {e}"

            trace_calls.append({"name": name, "args": args, "result": result_text})
            messages.append(
                ToolMessage(content=str(result_text), tool_call_id=tool_call_id)
            )

    # Agent failed to terminate — conservative fallback
    fallback = ClassificationResult(
        toxicity_score=0.5,
        reasoning="Agent exceeded max iterations without calling submit_verdict.",
        risk_category="Unclear",
        emoji_analysis=[],
    )
    return AgentResult(
        classification=fallback,
        trace=AgentTrace(
            iterations=max_iterations,
            tool_calls=trace_calls,
            terminated_normally=False,
        ),
    )
