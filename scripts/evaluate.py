"""CLI: Run evaluation suite.

Usage:
    python -m scripts.evaluate                     # full HatemojiCheck, 1 seed, 1k bootstrap
    python -m scripts.evaluate --sample-size 500   # stratified subsample
    python -m scripts.evaluate --n-seeds 3         # LLM stochasticity via mean ± std
    python -m scripts.evaluate --skip-hatemoji     # adversarial only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emoji_toxicity.evaluation.run_eval import run_full_evaluation


def main():
    p = argparse.ArgumentParser(description="Run emoji toxicity evaluation benchmarks")
    p.add_argument("--skip-hatemoji", action="store_true", help="Skip HatemojiCheck dataset")
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Stratified-sample HatemojiCheck to this size. Default: use full dataset.",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of LLM runs with different seeds for stochasticity estimation.",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resample count for 95%% CIs. 0 disables.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for stratified HatemojiCheck subsampling.",
    )
    p.add_argument(
        "--compare-modes",
        action="store_true",
        help="Also evaluate the fixed workflow (retrieve→classify) alongside the agent.",
    )
    p.add_argument(
        "--bench",
        action="store_true",
        help="Use the 155-sample context-flip benchmark instead of HatemojiBuild/adversarial. "
             "Tests context-sensitivity specifically (same emoji, different context → different label).",
    )
    args = p.parse_args()

    run_full_evaluation(
        include_hatemoji=not args.skip_hatemoji,
        sample_size=args.sample_size,
        n_seeds=args.n_seeds,
        n_bootstrap=args.n_bootstrap,
        sample_seed=args.sample_seed,
        compare_modes=args.compare_modes,
        use_context_flip_bench=args.bench,
    )


if __name__ == "__main__":
    main()
