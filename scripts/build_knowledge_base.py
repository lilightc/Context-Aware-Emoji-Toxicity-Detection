"""CLI: Build the unified knowledge base from all sources.

Usage:
    python -m scripts.build_knowledge_base [--skip-gpt] [--max-enrich N]
"""

from __future__ import annotations

import argparse
import sys

# Ensure project root is on path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emoji_toxicity.ingestion.build_kb import build_knowledge_base


def main():
    parser = argparse.ArgumentParser(description="Build unified emoji toxicity knowledge base")
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT enrichment")
    parser.add_argument("--max-enrich", type=int, default=None, help="Cap GPT enrichment calls")
    parser.add_argument("--include-urban", action="store_true", help="Include Urban Dictionary scraping")
    parser.add_argument("--include-silent-signals", action="store_true", help="Include Silent Signals dataset")
    args = parser.parse_args()

    build_knowledge_base(
        skip_urban=not args.include_urban,
        skip_silent_signals=not args.include_silent_signals,
        enrich_missing=not args.skip_gpt,
        max_enrich=args.max_enrich,
    )


if __name__ == "__main__":
    main()
