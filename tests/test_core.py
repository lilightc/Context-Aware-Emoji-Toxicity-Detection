"""Core unit tests for emoji toxicity detection components."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_cldr_loading():
    """CLDR loader returns emoji entries with expected fields."""
    from emoji_toxicity.ingestion.cldr import load_cldr_entries

    entries = load_cldr_entries()
    assert len(entries) > 100
    entry = entries[0]
    assert "symbol" in entry
    assert "literal_meaning" in entry
    assert entry["source"] == "CLDR"


def test_query_expansion():
    """Query expansion adds CLDR names to emoji."""
    from emoji_toxicity.detector.retriever import _expand_query
    from emoji_toxicity.utils import extract_emojis

    expanded = _expand_query("I love 🌽")
    assert "corn" in expanded.lower() or "maize" in expanded.lower()

    emojis = extract_emojis("Hello 🌽🍑 world")
    assert len(emojis) == 2
    assert "🌽" in emojis
    assert "🍑" in emojis


def test_keyword_baseline():
    """Keyword baseline flags known toxic emoji with monotone toxicity_score."""
    from emoji_toxicity.evaluation.baselines import keyword_baseline

    result_toxic = keyword_baseline("Nice 🍆")
    result_safe = keyword_baseline("Hello world!")
    assert result_toxic.toxicity_score > result_safe.toxicity_score
    assert result_toxic.toxicity_score >= 0.5
    assert result_safe.toxicity_score < 0.5


def test_adversarial_dataset():
    """Adversarial test set loads with expected structure."""
    from emoji_toxicity.evaluation.datasets import load_adversarial_test_set

    samples = load_adversarial_test_set()
    assert len(samples) >= 10

    # Check that we have both toxic and safe examples
    labels = {s.label for s in samples}
    assert 0 in labels
    assert 1 in labels

    # Check context-flip pairs exist
    sources = {s.source for s in samples}
    assert "adversarial" in sources


def test_metrics_computation():
    """Metrics computation returns expected fields."""
    from emoji_toxicity.evaluation.metrics import compute_metrics

    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]

    metrics = compute_metrics(y_true, y_pred, n_bootstrap=0)
    assert 0 <= metrics.accuracy.value <= 1
    assert 0 <= metrics.f1_macro.value <= 1
    assert len(metrics.confusion) == 2
    assert metrics.n_samples == 5


def test_bootstrap_ci():
    """Bootstrap CI returns bounds that contain the point estimate."""
    from emoji_toxicity.evaluation.metrics import compute_metrics

    # A modestly sized sample so the CI is meaningful
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5  # 50 samples, balanced
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1] * 5  # 80% accurate
    y_scores = [0.9, 0.1, 0.8, 0.2, 0.85, 0.15, 0.95, 0.05, 0.3, 0.7] * 5

    m = compute_metrics(y_true, y_pred, y_scores, n_bootstrap=200, rng_seed=42)

    # Point estimate lies within CI
    for metric in [m.accuracy, m.precision, m.recall, m.f1_macro]:
        assert metric.ci_low is not None
        assert metric.ci_high is not None
        assert metric.ci_low <= metric.value <= metric.ci_high
    # AUROC should also have a CI on this sample
    assert m.auroc is not None
    assert m.auroc.ci_low <= m.auroc.value <= m.auroc.ci_high

    # Summary string should contain CI brackets
    assert "[" in m.summary()


def test_stratified_sample():
    """Stratified sample preserves class presence and size."""
    from emoji_toxicity.evaluation.datasets import EvalSample, stratified_sample

    samples = [
        EvalSample(text=f"t{i}", context="", label=(i % 2), source="test")
        for i in range(100)
    ]

    picked = stratified_sample(samples, n=20, seed=42)
    assert len(picked) == 20
    labels = [s.label for s in picked]
    # Both classes should be represented
    assert 0 in labels and 1 in labels
    # Near-balanced
    assert abs(labels.count(0) - labels.count(1)) <= 2

    # n >= total: returns everything
    assert len(stratified_sample(samples, n=500)) == 100

    # Reproducible
    a = stratified_sample(samples, n=10, seed=7)
    b = stratified_sample(samples, n=10, seed=7)
    assert [s.text for s in a] == [s.text for s in b]


def test_config_loads():
    """Config loads without error."""
    from emoji_toxicity.config import settings, KB_PATH, DATA_DIR

    assert settings.embedding_dim == 384
    assert settings.retrieval_k == 3
    assert settings.detector_mode in ("agent", "workflow")
    assert KB_PATH.name == "knowledge_base.jsonl"


def test_agent_tool_schemas():
    """Agent tools are wired with the expected names and terminal marker."""
    from emoji_toxicity.detector.tools import AGENT_TOOLS, TERMINAL_TOOL_NAME

    names = {t.name for t in AGENT_TOOLS}
    assert names == {
        "lookup_emoji_knowledge",
        "search_similar_cases",
        "get_cldr_name",
        "submit_verdict",
    }
    assert TERMINAL_TOOL_NAME in names


def test_get_cldr_name_tool_offline():
    """get_cldr_name is a local lookup — runs without network."""
    from emoji_toxicity.detector.tools import get_cldr_name

    result = get_cldr_name.invoke({"emoji": "🌽"})
    assert "corn" in result.lower() or "maize" in result.lower()


def test_submit_verdict_reconstruction():
    """Verdict tool args reconstruct a valid ClassificationResult."""
    from emoji_toxicity.detector.tools import as_classification_result

    args = {
        "toxicity_score": 0.85,
        "reasoning": "Sexual context",
        "risk_category": "Sexual",
        "emoji_analysis": [
            {"emoji": "🌽", "interpretation": "slang for porn", "risk": "high"}
        ],
    }
    result = as_classification_result(args)
    assert result.toxicity_score == 0.85
    assert len(result.emoji_analysis) == 1
    assert result.emoji_analysis[0].risk == "high"


def test_score_to_verdict_gate():
    """Score gate maps scores to verdicts monotonically."""
    from emoji_toxicity.utils import verdict_from_score

    assert verdict_from_score(0.9) == "TOXIC"
    assert verdict_from_score(0.5) == "UNCERTAIN"
    assert verdict_from_score(0.1) == "SAFE"
    assert verdict_from_score(0.7) == "TOXIC"
    assert verdict_from_score(0.3) == "SAFE"
    # Custom thresholds
    assert verdict_from_score(0.6, toxic_threshold=0.6) == "TOXIC"
    assert verdict_from_score(0.4, safe_threshold=0.4) == "SAFE"


def test_context_flip_bench():
    """Context-flip benchmark loads with expected shape and balance."""
    from emoji_toxicity.evaluation.context_flip_bench import load_context_flip_bench, bench_stats

    samples = load_context_flip_bench()
    stats = bench_stats()
    assert stats["total"] >= 150
    assert abs(stats["toxic"] - stats["safe"]) <= 5  # near-balanced
    assert "hard" in stats["by_difficulty"]
    assert stats["by_difficulty"]["hard"] >= 50  # majority hard
    # All perturbation_types should be "difficulty:category"
    for s in samples:
        assert ":" in s.perturbation_type


def test_retrieval_gate():
    """Retrieval gate fires for coded emoji, skips for safe-only."""
    from emoji_toxicity.detector.retrieval_gate import needs_retrieval

    # Should trigger: coded emoji present
    assert needs_retrieval("She is a 🌽 star", "Check out my content") is True
    assert needs_retrieval("Got ❄️ tonight", "HMU") is True

    # Should trigger: slang context keywords even with safe emoji
    assert needs_retrieval("Check 😊", "link in bio") is True

    # Should NOT trigger: common safe emoji in benign context
    assert needs_retrieval("Great job! 👍", "Team meeting notes") is False
    assert needs_retrieval("Happy birthday! 🎂🎉", "Party at 5pm") is False

    # Should NOT trigger: no emoji at all
    assert needs_retrieval("Hello world", "") is False


def test_format_docs_with_scores():
    """format_retrieved_docs includes relevance scores when provided."""
    from emoji_toxicity.utils import format_retrieved_docs
    from langchain_core.documents import Document

    docs = [Document(page_content="test", metadata={
        "symbol": "🌽", "slang_meaning": "porn", "risk_category": "Sexual",
        "toxic_signals": "adult", "benign_signals": "farm"
    })]
    text_with, _ = format_retrieved_docs(docs, scores=[0.92])
    assert "0.92" in text_with  # score is included

    text_without, _ = format_retrieved_docs(docs)
    assert "Relevance" not in text_without  # no score


def test_calibrate_sweep():
    """Threshold calibration returns valid thresholds."""
    from scripts.calibrate_thresholds import calibrate

    # Synthetic: scores perfectly separate classes
    scores = [0.1, 0.2, 0.15, 0.9, 0.85, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    tt, st, f1 = calibrate(scores, labels, n_steps=20)
    assert st < tt
    assert f1 > 0.9  # should be perfect or near-perfect


def test_validation_gate():
    """Validation gate filters by confidence and source count."""
    from emoji_toxicity.ingestion.validation import validate_candidates

    candidates = [
        # High confidence, 3 sources → should pass
        {"symbol": "🧊", "slang_meaning": "ICE enforcement", "risk_category": "Political",
         "confidence": 0.9, "source_posts": [
             {"source": "reddit"}, {"source": "twitter"}, {"source": "emojipedia"}
         ]},
        # Low confidence → rejected
        {"symbol": "🫠", "slang_meaning": "cooked", "risk_category": "Safe",
         "confidence": 0.3, "source_posts": [{"source": "reddit"}]},
        # High confidence but only 1 source → rejected
        {"symbol": "🛣️", "slang_meaning": "promiscuity", "risk_category": "Sexual",
         "confidence": 0.95, "source_posts": [{"source": "tiktok"}]},
    ]

    result = validate_candidates(candidates, min_sources=3, min_confidence=0.7)
    assert len(result.accepted) == 1
    assert result.accepted[0]["symbol"] == "🧊"
    assert len(result.rejected) == 2


def test_kb_health_report():
    """KB health report runs without error (even with no KB)."""
    from emoji_toxicity.ingestion.monitor import kb_health_report

    report = kb_health_report()
    assert isinstance(report, dict)
    # Either "error" (no KB) or "total_entries" (has KB)
    assert "error" in report or "total_entries" in report


def test_detector_rejects_unknown_mode():
    """ToxicityDetector validates the mode argument."""
    import pytest
    from emoji_toxicity.detector.pipeline import ToxicityDetector

    with pytest.raises(ValueError):
        ToxicityDetector(mode="magic")

    # Adaptive mode should be accepted
    d = ToxicityDetector(mode="adaptive")
    assert d.mode == "adaptive"
