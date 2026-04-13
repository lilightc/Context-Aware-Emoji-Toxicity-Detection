"""End-to-end toxicity detection pipeline: text → DetectionResult."""

from __future__ import annotations

from dataclasses import dataclass

from emoji_toxicity.config import settings
from emoji_toxicity.detector.retriever import retrieve
from emoji_toxicity.detector.classifier import classify, ClassificationResult


@dataclass
class DetectionResult:
    """Final output of the toxicity detection pipeline."""
    verdict: str  # TOXIC, SAFE, or UNCERTAIN
    confidence: float
    reasoning: str
    risk_category: str
    emoji_analysis: list[dict]
    citations: list[str]  # Retrieved KB entries used
    raw_classification: ClassificationResult | None = None

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_category": self.risk_category,
            "emoji_analysis": self.emoji_analysis,
            "citations": self.citations,
        }


class ToxicityDetector:
    """End-to-end context-aware emoji toxicity detector using RAG.

    Pipeline: extract emoji → query expand → retrieve KB → LLM judge → gate confidence.
    """

    def __init__(
        self,
        retrieval_k: int | None = None,
        toxic_threshold: float | None = None,
        safe_threshold: float | None = None,
    ):
        self.retrieval_k = retrieval_k or settings.retrieval_k
        self.toxic_threshold = toxic_threshold or settings.toxic_threshold
        self.safe_threshold = safe_threshold or settings.safe_threshold

    def _format_retrieved_knowledge(self, documents) -> tuple[str, list[str]]:
        """Format retrieved documents into context string and citation list."""
        if not documents:
            return "No relevant emoji knowledge found.", []

        parts = []
        citations = []
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            symbol = meta.get("symbol", "?")
            entry_text = (
                f"[{i}] {symbol}\n"
                f"  Slang meaning: {meta.get('slang_meaning', 'N/A')}\n"
                f"  Risk category: {meta.get('risk_category', 'Unknown')}\n"
                f"  Toxic signals: {meta.get('toxic_signals', 'N/A')}\n"
                f"  Benign signals: {meta.get('benign_signals', 'N/A')}"
            )
            parts.append(entry_text)
            citations.append(f"{symbol}: {meta.get('slang_meaning', 'N/A')}")

        return "\n\n".join(parts), citations

    def _apply_confidence_gating(self, result: ClassificationResult) -> str:
        """Apply two-threshold confidence gating to produce final verdict.

        - confidence >= toxic_threshold → use LLM verdict (likely TOXIC)
        - confidence <= safe_threshold → SAFE
        - in between → UNCERTAIN
        """
        if result.confidence >= self.toxic_threshold:
            return result.verdict
        elif result.confidence <= self.safe_threshold:
            return "SAFE"
        else:
            return "UNCERTAIN"

    def detect(self, message: str, context: str = "") -> DetectionResult:
        """Run full detection pipeline on a message.

        Args:
            message: The message to analyze (may contain emoji).
            context: Optional surrounding context (parent post, thread, etc.).

        Returns:
            DetectionResult with verdict, confidence, reasoning, and citations.
        """
        retrieval = retrieve(message, k=self.retrieval_k)
        knowledge_text, citations = self._format_retrieved_knowledge(retrieval.documents)
        classification = classify(
            message=message,
            context_text=context or "No additional context provided.",
            retrieved_knowledge=knowledge_text,
        )
        final_verdict = self._apply_confidence_gating(classification)

        return DetectionResult(
            verdict=final_verdict,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            risk_category=classification.risk_category,
            emoji_analysis=[ea.model_dump() for ea in classification.emoji_analysis],
            citations=citations,
            raw_classification=classification,
        )
