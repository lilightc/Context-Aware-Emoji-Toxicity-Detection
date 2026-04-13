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
    """Keyword baseline flags known toxic emoji."""
    from emoji_toxicity.evaluation.baselines import keyword_baseline

    result_toxic = keyword_baseline("Nice 🍆")
    assert result_toxic.verdict == "TOXIC"

    result_safe = keyword_baseline("Hello world!")
    assert result_safe.verdict == "SAFE"


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

    metrics = compute_metrics(y_true, y_pred)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.f1_macro <= 1
    assert len(metrics.confusion) == 2


def test_config_loads():
    """Config loads without error."""
    from emoji_toxicity.config import settings, KB_PATH, DATA_DIR

    assert settings.embedding_dim == 384
    assert settings.retrieval_k == 3
    assert KB_PATH.name == "knowledge_base.jsonl"
