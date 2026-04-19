"""CLI: Calibrate toxic/safe thresholds on a held-out validation split.

Usage:
    python -m scripts.calibrate_thresholds [--val-ratio 0.35] [--n-steps 50]

Splits the context-flip benchmark into val (for sweep) and test (for final
report). Sweeps threshold pairs on val to maximize F1, then reports the best
thresholds and the test-set metrics at that operating point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emoji_toxicity.evaluation.context_flip_bench import load_context_flip_bench
from emoji_toxicity.evaluation.datasets import stratified_sample
from emoji_toxicity.evaluation.metrics import compute_metrics
from emoji_toxicity.utils import verdict_from_score


def calibrate(
    scores: list[float],
    labels: list[int],
    n_steps: int = 50,
) -> tuple[float, float, float]:
    """Sweep (toxic_threshold, safe_threshold) pairs on val data.

    Returns (best_toxic_thresh, best_safe_thresh, best_f1).
    """
    best_f1, best_tt, best_st = 0.0, 0.7, 0.3

    thresholds = np.linspace(0.05, 0.95, n_steps)
    for tt in thresholds:
        for st in thresholds:
            if st >= tt:
                continue  # safe_threshold must be < toxic_threshold
            preds = [
                0 if verdict_from_score(s, tt, st) == "SAFE" else 1
                for s in scores
            ]
            m = compute_metrics(labels, preds, n_bootstrap=0)
            if m.f1_macro.value > best_f1:
                best_f1 = m.f1_macro.value
                best_tt = float(tt)
                best_st = float(st)

    return best_tt, best_st, best_f1


def main():
    p = argparse.ArgumentParser(description="Calibrate score-gate thresholds")
    p.add_argument("--val-ratio", type=float, default=0.35,
                   help="Fraction of context-flip bench used for validation (rest = test)")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Number of threshold values to sweep per axis")
    p.add_argument("--scores-file", type=str, default=None,
                   help="Path to pre-computed scores JSONL (skips re-running the detector). "
                        "Each line: {\"text\": ..., \"label\": ..., \"toxicity_score\": ...}")
    args = p.parse_args()

    samples = load_context_flip_bench()
    n_val = int(len(samples) * args.val_ratio)
    n_test = len(samples) - n_val

    val_samples = stratified_sample(samples, n_val, seed=42)
    val_texts = {s.text for s in val_samples}
    test_samples = [s for s in samples if s.text not in val_texts]

    print(f"Val: {len(val_samples)} samples, Test: {len(test_samples)} samples\n")

    if args.scores_file:
        import json
        with open(args.scores_file) as f:
            records = [json.loads(line) for line in f if line.strip()]
        score_map = {r["text"]: r["toxicity_score"] for r in records}
        val_scores = [score_map[s.text] for s in val_samples if s.text in score_map]
        val_labels = [s.label for s in val_samples if s.text in score_map]
        test_scores = [score_map[s.text] for s in test_samples if s.text in score_map]
        test_labels = [s.label for s in test_samples if s.text in score_map]
    else:
        from emoji_toxicity.detector.pipeline import ToxicityDetector
        from tqdm import tqdm

        detector = ToxicityDetector(mode="workflow")
        print("Running workflow detector on all samples...")

        all_scores = {}
        for s in tqdm(samples, desc="Scoring"):
            r = detector.detect(s.text, s.context)
            all_scores[s.text] = r.toxicity_score

        val_scores = [all_scores[s.text] for s in val_samples]
        val_labels = [s.label for s in val_samples]
        test_scores = [all_scores[s.text] for s in test_samples]
        test_labels = [s.label for s in test_samples]

    # Sweep on val
    print("Sweeping thresholds on val set...")
    best_tt, best_st, val_f1 = calibrate(val_scores, val_labels, n_steps=args.n_steps)

    print(f"\nBest thresholds (val F1={val_f1:.3f}):")
    print(f"  TOXIC_THRESHOLD = {best_tt:.3f}")
    print(f"  SAFE_THRESHOLD  = {best_st:.3f}")

    # Evaluate on held-out test
    test_preds = [
        0 if verdict_from_score(s, best_tt, best_st) == "SAFE" else 1
        for s in test_scores
    ]
    test_m = compute_metrics(test_labels, test_preds, test_scores, n_bootstrap=1000)

    # Compare with default thresholds
    default_preds = [
        0 if verdict_from_score(s, 0.7, 0.3) == "SAFE" else 1
        for s in test_scores
    ]
    default_m = compute_metrics(test_labels, default_preds, test_scores, n_bootstrap=1000)

    print(f"\n{'':=<60}")
    print(f"Test set results (n={len(test_labels)}):")
    print(f"\n  Default thresholds (0.7 / 0.3):")
    print(f"  {default_m.summary()}")
    print(f"\n  Calibrated thresholds ({best_tt:.3f} / {best_st:.3f}):")
    print(f"  {test_m.summary()}")

    print(f"\nTo apply, set in .env:")
    print(f"  TOXIC_THRESHOLD={best_tt:.3f}")
    print(f"  SAFE_THRESHOLD={best_st:.3f}")


if __name__ == "__main__":
    main()
