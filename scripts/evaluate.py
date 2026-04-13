"""CLI: Run evaluation suite.

Usage:
    python -m scripts.evaluate [--max-hatemoji N] [--skip-hatemoji]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emoji_toxicity.evaluation.run_eval import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run emoji toxicity evaluation benchmarks")
    parser.add_argument("--max-hatemoji", type=int, default=200, help="Cap HatemojiCheck samples")
    parser.add_argument("--skip-hatemoji", action="store_true", help="Skip HatemojiCheck dataset")
    args = parser.parse_args()

    run_full_evaluation(
        include_hatemoji=not args.skip_hatemoji,
        max_hatemoji=args.max_hatemoji,
    )


if __name__ == "__main__":
    main()
