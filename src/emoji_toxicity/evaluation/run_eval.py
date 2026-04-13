"""Run all evaluation benchmarks and save results."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable

from tqdm import tqdm

from emoji_toxicity.config import RESULTS_DIR
from emoji_toxicity.detector.pipeline import ToxicityDetector
from emoji_toxicity.evaluation.datasets import (
    EvalSample,
    load_hatemoji_check,
    load_adversarial_test_set,
)
from emoji_toxicity.evaluation.baselines import keyword_baseline, raw_llm_baseline
from emoji_toxicity.evaluation.metrics import compute_metrics


def _verdict_to_label(verdict: str) -> int:
    """Convert verdict string to binary label. UNCERTAIN maps to TOXIC (conservative)."""
    return 0 if verdict == "SAFE" else 1


def _evaluate(
    samples: list[EvalSample],
    classify_fn: Callable[[EvalSample], tuple[str, float]],
    name: str,
    show_progress: bool = True,
) -> tuple[dict, list[tuple[EvalSample, str]]]:
    """Run a classifier function over samples and compute metrics.

    Returns (result_dict, per_sample_predictions) so callers can reuse predictions.
    """
    y_true, y_pred, y_scores = [], [], []
    per_sample: list[tuple[EvalSample, str]] = []

    iterator = tqdm(samples, desc=f"Eval ({name})") if show_progress else samples
    for sample in iterator:
        try:
            verdict, confidence = classify_fn(sample)
        except Exception as e:
            print(f"  [ERROR] {sample.text[:50]}... -> {e}")
            continue
        per_sample.append((sample, verdict))
        y_true.append(sample.label)
        y_pred.append(_verdict_to_label(verdict))
        y_scores.append(confidence)

    metrics = compute_metrics(y_true, y_pred, y_scores)
    return {"name": name, "metrics": metrics, "n_samples": len(y_true)}, per_sample


def run_full_evaluation(
    include_hatemoji: bool = True,
    max_hatemoji: int | None = 200,
) -> list[dict]:
    """Run all benchmarks and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    adversarial = load_adversarial_test_set()
    print(f"Loaded {len(adversarial)} adversarial test samples")

    all_samples = adversarial[:]
    if include_hatemoji:
        hatemoji = load_hatemoji_check()
        if max_hatemoji:
            hatemoji = hatemoji[:max_hatemoji]
        print(f"Loaded {len(hatemoji)} HatemojiCheck samples")
        all_samples.extend(hatemoji)

    print(f"Total evaluation samples: {len(all_samples)}\n")

    detector = ToxicityDetector()

    def kw_fn(s: EvalSample) -> tuple[str, float]:
        r = keyword_baseline(s.text, s.context)
        return r.verdict, r.confidence

    def llm_fn(s: EvalSample) -> tuple[str, float]:
        r = raw_llm_baseline(s.text, s.context)
        return r.verdict, r.confidence

    def rag_fn(s: EvalSample) -> tuple[str, float]:
        r = detector.detect(s.text, s.context)
        return r.verdict, r.confidence

    results = []
    rag_predictions: list[tuple[EvalSample, str]] = []

    for label, fn in [
        ("keyword_baseline", kw_fn),
        ("raw_llm", llm_fn),
        ("rag_pipeline", rag_fn),
    ]:
        print("=" * 60)
        print(f"Running {label}...")
        result, per_sample = _evaluate(all_samples, fn, label)
        print(result["metrics"].summary())
        results.append(result)
        if label == "rag_pipeline":
            rag_predictions = per_sample

    # Save results
    output = []
    for r in results:
        m = r["metrics"]
        output.append({
            "name": r["name"],
            "n_samples": r["n_samples"],
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1_macro": m.f1_macro,
            "auroc": m.auroc,
            "confusion_matrix": m.confusion,
        })

    results_path = RESULTS_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Per-perturbation breakdown — reuse RAG predictions, no extra inference.
    if adversarial:
        print("\n" + "=" * 60)
        print("Adversarial set breakdown by perturbation type:")
        adversarial_set = {id(s) for s in adversarial}
        by_type: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        for sample, verdict in rag_predictions:
            if id(sample) not in adversarial_set:
                continue
            pt = sample.perturbation_type or "unknown"
            by_type[pt]["total"] += 1
            if _verdict_to_label(verdict) == sample.label:
                by_type[pt]["correct"] += 1
        for pt, counts in sorted(by_type.items()):
            total = counts["total"]
            acc = counts["correct"] / total if total > 0 else 0
            print(f"  {pt}: {counts['correct']}/{total} ({acc:.0%})")

    return results
