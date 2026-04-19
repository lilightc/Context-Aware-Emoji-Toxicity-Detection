"""End-to-end toxicity detection pipeline: text → DetectionResult.

Two modes:
- ``agent`` (default): tool-calling agent chooses what/whether to retrieve
- ``workflow``: fixed retrieve → classify → gate pipeline (baseline for A/B)

The confidence gate derives the final TOXIC/SAFE/UNCERTAIN verdict purely from
the model's toxicity_score ∈ [0, 1]:
  score >= toxic_threshold  → TOXIC
  score <= safe_threshold   → SAFE
  in-between                → UNCERTAIN
"""

from __future__ import annotations

from dataclasses import dataclass

from emoji_toxicity.config import settings
from emoji_toxicity.detector.agent import AgentTrace, run_agent
from emoji_toxicity.detector.classifier import ClassificationResult, classify
from emoji_toxicity.detector.retrieval_gate import needs_retrieval
from emoji_toxicity.detector.retriever import retrieve
from emoji_toxicity.utils import format_retrieved_docs, verdict_from_score


@dataclass
class DetectionResult:
    """Final output of the toxicity detection pipeline."""
    verdict: str  # TOXIC, SAFE, or UNCERTAIN — derived from toxicity_score
    toxicity_score: float  # 0.0-1.0, the calibrated P(toxic) from the model
    reasoning: str
    risk_category: str
    emoji_analysis: list[dict]
    citations: list[str]
    mode: str = "agent"
    agent_trace: AgentTrace | None = None
    raw_classification: ClassificationResult | None = None

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "toxicity_score": self.toxicity_score,
            "reasoning": self.reasoning,
            "risk_category": self.risk_category,
            "emoji_analysis": self.emoji_analysis,
            "citations": self.citations,
            "mode": self.mode,
            "n_agent_iterations": self.agent_trace.iterations if self.agent_trace else None,
            "n_info_tool_calls": (
                sum(1 for c in self.agent_trace.tool_calls if c["name"] != "submit_verdict")
                if self.agent_trace
                else None
            ),
        }


class ToxicityDetector:
    """Context-aware emoji toxicity detector.

    ``mode="agent"`` uses a tool-calling GPT-5 agent that decides whether and
    what to retrieve. ``mode="workflow"`` runs the fixed retrieve→classify
    pipeline (useful as an ablation baseline).
    """

    def __init__(
        self,
        mode: str | None = None,
        retrieval_k: int | None = None,
        toxic_threshold: float | None = None,
        safe_threshold: float | None = None,
        agent_max_iterations: int | None = None,
    ):
        self.mode = mode or settings.detector_mode
        if self.mode not in ("agent", "workflow", "adaptive"):
            raise ValueError(
                f"Unknown mode {self.mode!r}; expected 'agent', 'workflow', or 'adaptive'."
            )
        self.retrieval_k = retrieval_k or settings.retrieval_k
        self.toxic_threshold = toxic_threshold or settings.toxic_threshold
        self.safe_threshold = safe_threshold or settings.safe_threshold
        self.agent_max_iterations = agent_max_iterations or settings.agent_max_iterations

    # ---------- workflow mode (baseline) ----------

    def _detect_workflow(
        self, message: str, context: str, seed: int | None
    ) -> DetectionResult:
        retrieval = retrieve(message, k=self.retrieval_k)
        knowledge_text, citations = format_retrieved_docs(
            retrieval.documents, scores=retrieval.scores
        )
        classification = classify(
            message=message,
            context_text=context or "No additional context provided.",
            retrieved_knowledge=knowledge_text,
            seed=seed,
        )
        return DetectionResult(
            verdict=verdict_from_score(classification.toxicity_score, self.toxic_threshold, self.safe_threshold),
            toxicity_score=classification.toxicity_score,
            reasoning=classification.reasoning,
            risk_category=classification.risk_category,
            emoji_analysis=[ea.model_dump() for ea in classification.emoji_analysis],
            citations=citations,
            mode="workflow",
            raw_classification=classification,
        )

    # ---------- agent mode (default) ----------

    @staticmethod
    def _citations_from_trace(trace: AgentTrace) -> list[str]:
        out = []
        for call in trace.tool_calls:
            name = call["name"]
            if name in ("lookup_emoji_knowledge", "search_similar_cases"):
                key = call["args"].get("emoji") or call["args"].get("query", "?")
                out.append(f"{name}({key})")
            elif name == "get_cldr_name":
                out.append(f"get_cldr_name({call['args'].get('emoji', '?')})")
        return out

    def _detect_agent(
        self, message: str, context: str, seed: int | None
    ) -> DetectionResult:
        agent_result = run_agent(
            message=message,
            context=context,
            seed=seed,
            max_iterations=self.agent_max_iterations,
        )
        classification = agent_result.classification
        return DetectionResult(
            verdict=verdict_from_score(classification.toxicity_score, self.toxic_threshold, self.safe_threshold),
            toxicity_score=classification.toxicity_score,
            reasoning=classification.reasoning,
            risk_category=classification.risk_category,
            emoji_analysis=[ea.model_dump() for ea in classification.emoji_analysis],
            citations=self._citations_from_trace(agent_result.trace),
            mode="agent",
            agent_trace=agent_result.trace,
            raw_classification=classification,
        )

    # ---------- adaptive mode ----------

    def _detect_adaptive(
        self, message: str, context: str, seed: int | None
    ) -> DetectionResult:
        """Use the lightweight retrieval gate to decide whether to retrieve.

        If the gate fires → full workflow (retrieve + classify).
        If not → classify without retrieval (like raw_llm but with structured prompt).
        """
        if needs_retrieval(message, context):
            return self._detect_workflow(message, context, seed)

        # No retrieval — classify directly with "no relevant entries" context
        classification = classify(
            message=message,
            context_text=context or "No additional context provided.",
            retrieved_knowledge="No retrieval performed — message appears to use emoji literally.",
            seed=seed,
        )
        return DetectionResult(
            verdict=verdict_from_score(
                classification.toxicity_score, self.toxic_threshold, self.safe_threshold
            ),
            toxicity_score=classification.toxicity_score,
            reasoning=classification.reasoning,
            risk_category=classification.risk_category,
            emoji_analysis=[ea.model_dump() for ea in classification.emoji_analysis],
            citations=["(retrieval skipped by adaptive gate)"],
            mode="adaptive",
            raw_classification=classification,
        )

    # ---------- entry point ----------

    def detect(
        self, message: str, context: str = "", seed: int | None = None
    ) -> DetectionResult:
        if self.mode == "agent":
            return self._detect_agent(message, context, seed)
        if self.mode == "adaptive":
            return self._detect_adaptive(message, context, seed)
        return self._detect_workflow(message, context, seed)
